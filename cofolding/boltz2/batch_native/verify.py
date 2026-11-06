"""Saved-byte Boltz2 gates extracted/generalized from verify_job/verify_independent.
Only trusted run-owned torch/pickle artifacts may be read. No model loads.
"""
import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
import gemmi
import numpy as np
import torch
from rdkit import Chem
from graph import graph_check, require
from runtime import native_args
SCALARS=('confidence_score','ptm','iptm','ligand_iptm','protein_iptm','complex_plddt','complex_iplddt','complex_pde','complex_ipde')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def eq(a,b,label): require(np.array_equal(a,b),label)


def packet(path):
    obj=torch.load(path,map_location='cpu',weights_only=False); inventory={}
    def walk(v,k=''):
        if torch.is_tensor(v):
            finite=bool(torch.isfinite(v).all()) if v.is_floating_point() or v.is_complex() else True
            require(finite,'nonfinite '+k);inventory[k]={'shape':list(v.shape),'dtype':str(v.dtype),'finite':finite}
        elif isinstance(v,dict):
            for key,val in v.items():walk(val,k+'/'+str(key))
        elif isinstance(v,(list,tuple)):
            for i,val in enumerate(v):walk(val,k+'/'+str(i))
        elif isinstance(v,float): require(np.isfinite(v),'nonfinite scalar')
    walk(obj)
    require(inventory==json.loads(path.with_name(path.stem+'_inventory.json').read_text()),'tensor inventory')
    return obj


def check_input(data,processed,rid,b):
    from boltz.data.types import StructureV2,MSA
    from boltz.data import const
    from boltz.data.parse.a3m import parse_a3m
    s=StructureV2.load(processed/f'structures/{rid}.npz')
    entities={e[next(iter(e))]['id']:(next(iter(e)),e[next(iter(e))]) for e in data['sequences']}
    require({str(c['name']) for c in s.chains}==set(entities) and s.mask.all(),'all source chains')
    require(int(b['atom_pad_mask'].sum())==len(s.atoms),'full atom coverage')
    require(b['contact_pair_index'].numel()==0,'unexpected contacts')
    if 'template_force' in b: require(not b['template_force'].any(),'unexpected template')
    with (processed/f'mols/{rid}.pkl').open('rb') as f:mols=pickle.load(f)
    token_count=0; expected_elements={}
    manifest=json.loads((processed/'manifest.json').read_text())
    record=next(r for r in manifest['records'] if r['id']==rid)
    for c in s.chains:
        cid=str(c['name']);kind,e=entities[cid]
        rs=s.residues[int(c['res_idx']):int(c['res_idx']+c['res_num'])]
        if kind=='protein':
            eq(rs['res_type'],[const.token_ids[const.prot_letter_to_token[x]] for x in e['sequence']],'protein sequence')
            token_count+=len(e['sequence'])
            chain=next(x for x in record['chains'] if x['chain_name']==cid)
            if e['msa']=='empty':
                require(chain['msa_id']==-1,'explicit no MSA')
                require(b['msa'].shape[1]==1,'consumed single-sequence depth')
            else:
                fresh=parse_a3m(Path(e['msa']),None,8192);stored=MSA.load(processed/f"msa/{chain['msa_id']}.npz")
                for key in ('sequences','residues','deletions'):eq(getattr(fresh,key),getattr(stored,key),'MSA replay '+key)
                require(b['msa'].shape[1]==len(stored.sequences),'consumed MSA depth')
        else:
            require(len(rs)==1,'single ligand residue')
            source=Chem.MolFromSmiles(e['smiles']);require(source is not None,'invalid source SMILES')
            mol=Chem.RemoveHs(mols[str(rs[0]['name'])]);match=graph_check(source,mol)
            start=int(c['atom_idx']);count=int(c['atom_num']);require(count==source.GetNumAtoms(),'ligand atoms')
            names=[str(a['name']) for a in s.atoms[start:start+count]];mapping={}
            for i,j in enumerate(match):
                atom=mol.GetAtomWithIdx(j);idx=start+names.index(atom.GetProp('name'));mapping[i]=idx
                require(int(b['ref_element'][0,idx].argmax())==atom.GetAtomicNum(),'consumed element')
                require(float(b['ref_charge'][0,idx])==atom.GetFormalCharge(),'consumed charge')
                expected_elements[(cid,atom.GetProp('name'))]=atom.GetSymbol()
            bonds={tuple(sorted((mapping[x.GetBeginAtomIdx()],mapping[x.GetEndAtomIdx()]))) for x in source.GetBonds()}
            actual={tuple(sorted((int(x['atom_1']),int(x['atom_2'])))) for x in s.bonds if start<=int(x['atom_1'])<start+count and start<=int(x['atom_2'])<start+count}
            require(actual==bonds,'ligand connectivity');token_count+=count
    require(int(b['token_pad_mask'].sum())==token_count,'full tokens')
    eq(np.unique(b['asym_id'][b['token_pad_mask'].bool()].numpy()),np.arange(len(entities)),'consumed chains')
    return s,expected_elements


def check_cif(path,s,coords,elements):
    # Gemmi reads the native PDB or mmCIF directly; never convert coordinates.
    st=gemmi.read_structure(str(path));require(len(st)==1,'one structure model');actual={};expected={}
    for c in st[0]:
        for r in c:
            for a in r:
                key=(c.name,r.seqid.num,a.name);require(key not in actual,'duplicate atom')
                actual[key]=np.array([a.pos.x,a.pos.y,a.pos.z])
                if (c.name,a.name) in elements:require(a.element.name==elements[(c.name,a.name)],'structure ligand element')
    for c in s.chains:
        for r in s.residues[int(c['res_idx']):int(c['res_idx']+c['res_num'])]:
            for idx in range(int(r['atom_idx']),int(r['atom_idx']+r['atom_num'])):expected[(str(c['name']),int(r['res_idx'])+1,str(s.atoms[idx]['name']))]=coords[idx]
    require(set(actual)==set(expected),'structure full atom inventory')
    for k,v in expected.items():require(np.allclose(actual[k],v,rtol=0,atol=0.00051),'structure coordinate pairing')


def verify(root,seed):
    stage=root/f'seed-{seed}'
    saved=stage/'verified.json'
    prior=None
    if saved.exists():
        raw=saved.read_bytes()
        saved.unlink()  # A failed re-read must never leave an authoritative PASS.
        prior=json.loads(raw)
        for name,digest in prior['sha256'].items(): require(sha(stage/name)==digest,'modified output '+name)
    req=json.loads((root/'request.json').read_text());settings=req['settings'];n=settings['diffusion_samples']
    argv=json.loads((stage/'argv.json').read_text())
    require(argv==native_args(stage,seed,settings),'native command changed')
    events=[json.loads(line) for line in (stage/'events.jsonl').read_text().splitlines()]
    for name in ('load_start','load_end','cli_start','cli_end'):
        require(sum(e['event']==name for e in events)==1,'native lifecycle '+name)
    require(not any(e['event']=='cli_error' for e in events),'native CLI error')
    for name in ('predict_start','predict_end'):
        require(sorted(e['record'] for e in events if e['event']==name)==sorted(r['id'] for r in req['records']),'native record lifecycle '+name)
    native=stage/'native/boltz_results_inputs';processed=native/'processed';ranks={}
    require({p.name for p in (stage/'capture').iterdir()}=={r['id'] for r in req['records']},'capture record coverage')
    require({p.name for p in (native/'predictions').iterdir()}=={r['id'] for r in req['records']},'prediction record coverage')
    for rec in req['records']:
        rid=rec['id'];cap=stage/'capture'/rid
        b,f,p=[packet(cap/(label+'.pt')) for label in ('consumed_batch','full_forward','full_prediction')]
        require(p['exception'] is False and p['coords'].shape[0]==n,'samples/native exception')
        require({'pdistogram','s','z','pde_logits','pae_logits','plddt_logits','resolved_logits'}<=set(f),'full forward fields')
        require({'pae','pde','plddt','ptm','iptm','pair_chains_iptm','s','z'}<=set(p),'full prediction fields')
        eq(p['coords'],f['sample_atom_coords'],'forward coords');eq(p['masks'],b['atom_pad_mask'],'atom mask')
        s,elements=check_input(rec['yaml'],processed,rid,b)
        rank=torch.argsort(p['confidence_score'],descending=True).tolist();require(sorted(rank)==list(range(n)),'rank bijection');ranks[rid]=rank
        extension = 'pdb' if settings.get('output_format', 'mmcif') == 'pdb' else 'cif'
        pred=native/'predictions'/rid
        expected={'embeddings_'+rid+'.npz'} if settings.get('write_embeddings', True) else set()
        for r,i in enumerate(rank):
            base=f'{rid}_model_{r}';expected|={base+'.'+extension,'confidence_'+base+'.json',*{k+'_'+base+'.npz' for k in ('pae','pde','plddt')}}
            check_cif(pred/(base+'.'+extension),s,p['coords'][i][p['masks'][0].bool()].numpy(),elements)
            conf=json.loads((pred/('confidence_'+base+'.json')).read_text())
            for field in SCALARS:eq(conf[field],p[field][i].numpy(),'confidence '+field)
            for a,vs in p['pair_chains_iptm'].items():
                for c,v in vs.items():eq(conf['pair_chains_iptm'][str(a)][str(c)],v[i].numpy(),'chain confidence')
                eq(conf['chains_ptm'][str(a)],vs[a][i].numpy(),'chain ptm')
            for field in ('pae','pde','plddt'):
                with np.load(pred/(field+'_'+base+'.npz'),allow_pickle=False) as z:eq(z[field],p[field][i].numpy(),field+' pairing')
        require({p.name for p in pred.iterdir()}==expected,'exact native output inventory')
        if settings.get('write_embeddings', True):
            with np.load(pred/('embeddings_'+rid+'.npz'),allow_pickle=False) as z:
                require(set(z.files)=={'s','z'},'embeddings fields')
                for k in z.files:eq(z[k],p[k].numpy(),'embeddings preservation')
    result={'status':'PASS','models':len(req['records'])*n,'rank_to_original_sample':ranks,'sha256':{str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file() and p.name!='verified.json'}}
    if prior is not None:require(result==prior,'output inventory changed')
    # Keep the temporary receipt outside the inventoried seed directory.
    fd,tmp=tempfile.mkstemp(prefix=f'.verified-{seed}-',dir=root)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,saved)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    print(json.dumps({'status':'PASS','seed':seed,'models':result['models']}))

if __name__=='__main__':verify(Path('/run'),int(sys.argv[1]))
