"""Independent checks on saved Protenix artifacts; never trusts the engine's own report.

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
    from rdkit import Chem

    chains_expected = chemistry.get('chains') or []
    require(bool(chains_expected), 'chemistry.json carries no chain records')
    expected_chains = {chain['cif_chain'] for chain in chains_expected}
    protein_chains = {chain['cif_chain']: chain for chain in chains_expected
                      if chain['kind'] == 'protein'}
    ligand_chains = [chain for chain in chains_expected if chain['kind'] == 'ligand']
    protein_residues = chemistry['residues']
    ligand_records = chemistry.get('ligands') or []
    require(len(ligand_chains) == len(ligand_records),
            'chemistry.json ligand chain count does not match its ligand records')
    ligand_references = {}
    for chain, record in zip(ligand_chains, ligand_records):
        molecule = Chem.MolFromSmiles(record['consumed_smiles'])
        require(molecule is not None, 'consumed ligand SMILES not parseable')
        ligand_references[chain['cif_chain']] = molecule

    # Protenix PAE is indexed by tokens: one per protein residue, one per ligand atom.
    tokens = protein_residues + sum(mol.GetNumHeavyAtoms() for mol in ligand_references.values())
    confidence_files = sorted(native.glob('confidence_*.json'))
    require(len(confidence_files) == samples, 'confidence file count does not match samples')
    for index in range(samples):
        number = index + 1
        label = rid + '_sample_' + str(number)
        cif = native / (label + '.cif')
        require(cif.is_file(), 'missing native structure ' + cif.name)
        chains, atoms = parse_cif(cif)
        require(set(chains) == expected_chains,
                'chain set ' + str(sorted(set(chains))) + ' != ' + str(sorted(expected_chains)))
        coordinates = np.asarray([[atom[5], atom[6], atom[7]] for atom in atoms], dtype=np.float64)
        require(coordinates.size > 0 and np.isfinite(coordinates).all(), 'non-finite coordinates')

        protein = [atom for atom in atoms if atom[0] in protein_chains]
        require(len(protein) > 0, 'no protein atoms')
        residues = {(atom[0], atom[1]) for atom in protein}
        require(len(residues) == protein_residues,
                'protein residue count ' + str(len(residues)) + ' != ' + str(protein_residues))
        for chain_id, chain in protein_chains.items():
            observed = {atom[1] for atom in atoms if atom[0] == chain_id}
            require(len(observed) == chain['residues'],
                    'chain ' + chain_id + ' residue count ' + str(len(observed))
                    + ' != ' + str(chain['residues']))

        for chain_id, reference in ligand_references.items():
            observed = Counter(atom[4] for atom in atoms if atom[0] == chain_id)
            require(sum(observed.values()) == reference.GetNumHeavyAtoms(),
                    'ligand heavy-atom count mismatch: ' + str(sum(observed.values())))
            require(observed == heavy(reference),
                    'ligand element counts ' + str(dict(observed)) + ' != '
                    + str(dict(heavy(reference))))

        confidence = read_json(native / ('confidence_' + label + '.json'))
        require(confidence['residues'] == protein_residues,
                'confidence residue count ' + str(confidence['residues']) + ' != '
                + str(protein_residues))
        require(confidence['chains'] == [chain['cif_chain'] for chain in chains_expected],
                'confidence chain mapping does not match the requested entities')
        require(confidence['atoms'] == len(atoms),
                'confidence atom count ' + str(confidence['atoms']) + ' != written atoms '
                + str(len(atoms)))
        require(confidence['tokens'] == tokens,
                'confidence token count ' + str(confidence['tokens']) + ' != ' + str(tokens))
        with np.load(native / ('pae_' + label + '.npz'), allow_pickle=False) as arrays:
            pae = arrays['pae']
        require(pae.ndim == 2 and pae.shape[0] == pae.shape[1], 'PAE must be square')
        require(pae.shape == (tokens, tokens), 'PAE token coverage mismatch')
        require(np.isfinite(pae).all(), 'non-finite PAE')
        with np.load(native / ('plddt_' + label + '.npz'), allow_pickle=False) as arrays:
            plddt = arrays['plddt']
        require(plddt.shape == (len(atoms),), 'native atom pLDDT length mismatch')
        require(np.isfinite(plddt).all(), 'non-finite pLDDT')
        require(float(plddt.min()) >= 0.0 and float(plddt.max()) <= 100.0,
                'pLDDT outside the 0-100 native scale')
        with np.load(native / ('pde_' + label + '.npz'), allow_pickle=False) as arrays:
            pde = arrays['pde']
        require(pde.shape == (tokens, tokens), 'native PDE token coverage mismatch')
        require(np.isfinite(pde).all(), 'non-finite PDE')

        # The raw Protenix writer directory is optional evidence; check it when present.
        seed = prepared['sample_seeds'][index]
        raw = native / 'protenix' / (rid + '.seed_' + str(seed)) / rid / ('seed_' + str(seed))
        produced = raw / 'predictions'
        if produced.is_dir():
            require((produced / (rid + '_sample_0.cif')).is_file(),
                    'raw Protenix sample directory is missing its CIF')
            require((produced / (rid + '_summary_confidence_sample_0.json')).is_file(),
                    'raw Protenix sample directory is missing its summary confidence')
            full_data = produced / (rid + '_full_data_sample_0.json')
            require(full_data.is_file(),
                    'raw Protenix sample directory is missing its full-data confidence')
            # The writer's own copy of the PAE is the rounded token_pair_pae; comparing it with
            # the array we scored proves the scores used the engine's matrix, not a transcription.
            if full_data.stat().st_size <= 64 * 1024 * 1024:
                written = np.asarray(read_json(full_data)['token_pair_pae'], dtype=float)
                require(written.shape == (tokens, tokens), 'native confidence PAE shape mismatch')
                require(np.max(np.abs(written - np.round(pae.astype(float), 2))) <= 0.011,
                        'written PAE does not match the native confidence output')

    require(sorted(p.name for p in native.glob('*_sample_*.cif')) ==
            sorted(rid + '_sample_' + str(i + 1) + '.cif' for i in range(samples)),
            'unexpected native structure inventory')
