"""Unmodified graph bijection gate extracted from verify_independent.py."""
from rdkit import Chem
def require(test,label):
 if not test: raise ValueError(label)
def graph_check(src,dst):
 require(Chem.MolToSmiles(src,isomericSmiles=True)==Chem.MolToSmiles(dst,isomericSmiles=True),'processed graph/charge/isotope/stereo mismatch')
 matches=dst.GetSubstructMatches(src,useChirality=True,uniquify=False,maxMatches=10000)
 matches=[x for x in matches if len(x)==dst.GetNumAtoms()==src.GetNumAtoms() and len(set(x))==len(x)]
 require(bool(matches),'full graph bijection');match=min(matches)
 for i,j in enumerate(match):
  a,b=src.GetAtomWithIdx(i),dst.GetAtomWithIdx(j)
  require((a.GetAtomicNum(),a.GetFormalCharge(),a.GetIsotope(),a.GetIsAromatic())==(b.GetAtomicNum(),b.GetFormalCharge(),b.GetIsotope(),b.GetIsAromatic()),'mapped atom state')
 for b in src.GetBonds():
  other=dst.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()],match[b.GetEndAtomIdx()]);require(other is not None and other.GetBondType()==b.GetBondType(),'mapped bond')
 return match
