"""Independent checks on saved ESMFold2 artifacts; never trusts the engine's own report.

Geometry, entity and confidence checks are recomputed from the written files, so a lost ligand,
a truncated protein or a swapped sample cannot pass by re-reading the engine's summary.
"""
from collections import Counter
import json
from pathlib import Path

import numpy as np

CIF_PRECISION = 5e-3  # mmCIF coordinates carry three decimals


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def parse_cif(path):
    import gemmi
    structure = gemmi.read_structure(str(path))
    require(len(structure) >= 1, 'empty CIF model')
    model = structure[0]
    chains = [chain.name for chain in model]
    atoms = [(chain.name, residue.seqid.num, residue.name, atom.name, atom.element.name,
              atom.pos.x, atom.pos.y, atom.pos.z)
             for chain in model for residue in chain for atom in residue]
    return chains, atoms


def heavy(mol):
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def verify_job(prepared, attempt, settings):
    attempt = Path(attempt)
    rid = prepared['rid']
    samples = settings.samples
    chemistry = read_json(attempt / 'chemistry.json')
    native = attempt / 'native'
    capture = attempt / 'capture'
    from rdkit import Chem

    expected_chains = set(prepared['entities'])
    protein_residues = chemistry['residues']
    ligand_reference = None
    if chemistry.get('ligand'):
        ligand_reference = Chem.MolFromSmiles(chemistry['ligand']['consumed_smiles'])
        require(ligand_reference is not None, 'consumed ligand SMILES not parseable')
    heme_reference = None
    if chemistry.get('heme'):
        heme_reference = Chem.MolFromSmiles(chemistry['heme']['consumed_smiles'])
        require(heme_reference is not None, 'consumed heme SMILES not parseable')

    # Native confidence is indexed by protein residues plus atom-expanded
    # ligand/heme tokens, not by protein residues alone.
    tokens = protein_residues + sum(mol.GetNumHeavyAtoms() for mol in
                                    (ligand_reference, heme_reference) if mol is not None)
    confidence_files = sorted(native.glob('confidence_*.json'))
    require(len(confidence_files) == samples, 'confidence file count does not match samples')
    for index in range(samples):
        label = rid + '_sample_' + str(index + 1)
        cif = native / (label + '.cif')
        require(cif.is_file(), 'missing native structure ' + cif.name)
        chains, atoms = parse_cif(cif)
        require(set(chains) == expected_chains,
                'chain set ' + str(sorted(set(chains))) + ' != ' + str(sorted(expected_chains)))
        coordinates = np.asarray([[atom[5], atom[6], atom[7]] for atom in atoms], dtype=np.float64)
        require(coordinates.size > 0 and np.isfinite(coordinates).all(), 'non-finite coordinates')

        protein = [atom for atom in atoms if atom[0] == 'A']
        require(len(protein) > 0, 'no protein atoms')
        residues = {atom[1] for atom in protein}
        require(len(residues) == protein_residues,
                'protein residue count ' + str(len(residues)) + ' != ' + str(protein_residues))

        if ligand_reference is not None:
            observed = Counter(atom[4] for atom in atoms if atom[0] == 'L')
            require(sum(observed.values()) == ligand_reference.GetNumHeavyAtoms(),
                    'ligand heavy-atom count mismatch: ' + str(sum(observed.values())))
            require(observed == heavy(ligand_reference),
                    'ligand element counts ' + str(dict(observed)) + ' != ' + str(dict(heavy(ligand_reference))))
        if heme_reference is not None:
            observed = Counter(atom[4] for atom in atoms if atom[0] == 'H')
            require(sum(observed.values()) == heme_reference.GetNumHeavyAtoms(),
                    'heme heavy-atom count mismatch')
            require(observed == heavy(heme_reference), 'heme element counts mismatch')

        confidence = read_json(native / ('confidence_' + label + '.json'))
        require(confidence['residues'] == protein_residues,
                'confidence residue count ' + str(confidence['residues']) + ' != ' + str(protein_residues))
        with np.load(native / ('pae_' + label + '.npz'), allow_pickle=False) as arrays:
            pae = arrays['pae']
        require(pae.ndim == 2 and pae.shape[0] == pae.shape[1], 'PAE must be square')
        require(pae.shape == (tokens, tokens), 'PAE token coverage mismatch')
        require(confidence.get('tokens') == tokens, 'confidence token count mismatch')
        require(np.isfinite(pae).all(), 'non-finite PAE')
        with np.load(native / ('plddt_' + label + '.npz'), allow_pickle=False) as arrays:
            plddt = arrays['plddt']
        require(plddt.shape == (tokens,), 'native token pLDDT length mismatch')
        require(np.isfinite(plddt).all(), 'non-finite pLDDT')

        packet = capture / ('full_result_sample_' + str(index + 1) + '.pt')
        require(packet.is_file(), 'missing captured native result')
        import torch
        result = torch.load(packet, map_location='cpu', weights_only=False)
        require(result.complex is not None, 'captured result has no complex')
        positions = np.asarray(result.complex.atom_positions, dtype=np.float64)
        require(positions.shape == coordinates.shape,
                'captured coordinate count ' + str(positions.shape) + ' != CIF ' + str(coordinates.shape))
        deviation = float(np.abs(positions - coordinates).max())
        require(deviation <= CIF_PRECISION,
                'captured coordinates differ from the written CIF by ' + str(deviation) + ' A')
        inventory = read_json(capture / ('full_result_sample_' + str(index + 1) + '_inventory.json'))
        require(bool(inventory), 'empty captured tensor inventory')
        for path, entry in inventory.items():
            require(entry['finite'], 'non-finite captured tensor ' + path)

    require(sorted(p.name for p in native.glob('*_sample_*.cif')) ==
            sorted(rid + '_sample_' + str(i + 1) + '.cif' for i in range(samples)),
            'unexpected native structure inventory')
