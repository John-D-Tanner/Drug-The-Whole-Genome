from pathlib import Path
from rdkit import Chem
import numpy as np
import lmdb
import pickle
from chembl_webresource_client.new_client import new_client
import MDAnalysis as mda


# sdf to mol
# get coords from mol
# get elements from mol
# get chembl id
# get smiles from where?

# read lmdb
# atom ids from lmdb


# Adapted from https://stackoverflow.com/questions/31649216/writing-data-to-lmdb-with-python-very-slow
# funciton must be used on an open lmdb
def write_to_lmdb(db, key, value):
    """
    Write (key,value) to db. 
    Updates map size if it is insufficent. Map size is pre-specified and defaults to 10MiB. An additional 10 MiB is added upon each failure
    """
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(str(key).encode(), pickle.dumps(value))
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit + 10485760 #This is 10MiB in bytes
            print(f'increasing lmdb map size to {new_limit / (1024 * 1024)}MiB ...')
            db.set_mapsize(new_limit)




def explore_lmdb(db_path):
    # 1. Open the environment in read-only mode
    env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, readahead=False)

    contents = []
    with env.begin() as txn:
        # 2. Create a cursor to iterate over the DB
        cursor = txn.cursor()

        # print(f"{'Key (PDB ID)':<20} | {'Data Type'}")
        # print("-" * 40)

        for i, (key, value) in enumerate(cursor):
            # Keys are stored as bytes, decode to string
            key_str = key.decode('utf-8')

            # Unpickle the value to see what's inside
            try:
                data = pickle.loads(value)
                # print(f"{key_str:<20} | {type(data)}")
                contents.append(data)

            except Exception as e:
                print(f"Could not decode value for {key_str}: {e}")

            # Safety: remove the break to see ALL keys
            # if i > 9:
            #     break
            # else:
            #     print(data['smi'])

    env.close()
    return contents

#Writes arbriary dict to lmdb. 
def dict_to_lmdb(d):
    with lmdb.open('Test/test2.lmdb', ) as opened_lmdb:
        for i, mol_dict in enumerate(d):
            write_to_lmdb(opened_lmdb, i, mol_dict)





class ProcessLigands():

    @classmethod
    def sdf_to_lmdb(cls, sdf, outlmdb_name,):
        """ Converts an sdf file which may contain more than one molecule into a single lmdb file in the drugclip format """
        
        # Get molecule attributes
        mols = cls.get_mols_from_sdf(sdf)
        print('mols in sdf: ', len(mols))
        coords = cls.get_coords(mols)
        elements = cls.get_elements(mols)
        chembl_ids = cls.get_chembl(mols)
        smiles_strings = cls.get_smiles(mols)

        print('prop lens: ', [len(i) for i in [coords, elements, chembl_ids, smiles_strings, mols]])


        # Format into dictionary
        mol_dicts = []
        for coord_arr, element_arr, chembl_id, smiles, mol in zip(coords, elements, chembl_ids, smiles_strings, mols):
            mol_dict = {
                'coordinates': coord_arr,
                'atoms': element_arr,
                'smi': f'{chembl_id}_{smiles}',
                'mol':mol,
                'label':'arbitrary_string', #Key error is given by drugclip if this is not here. But it doesn't seem to actually be used anywhere. There own lmdbs don't have it!
            }
            mol_dicts.append(mol_dict)
        print('nmols: ', len(mol_dicts))
        # Create pickle of dict and put into lmdb file (as required by drugclip)
        with lmdb.open(outlmdb_name, map_async=True) as opened_lmdb:
            for i, mol_dict in enumerate(mol_dicts):
                write_to_lmdb(opened_lmdb, i, mol_dict)


    @staticmethod
    def get_mols_from_sdf(sdf, ):
        """From an sdf file, returns rdkit MOl objects for every molecule therein"""
        return list(Chem.SDMolSupplier(sdf))

    @staticmethod
    def get_elements(mols):
        """For the inputted rdkit mol objects, return a list of numpy arrays of elements (no Hydrogen)""" #how to implement no H?
        elements = []
        for mol in mols:
            elements.append([atom.GetSymbol() for atom in mol.GetAtoms()])
        return elements

    @staticmethod
    def get_coords(mols):
        """For the inputted rdkit mol objects, return a list of numpy arrays of molecule coordinates"""
        return [mol.GetConformer().GetPositions() for mol in mols]

    @staticmethod
    def get_chembl(mols):
        """ From the ChEMBL database online, it retrieves the chembl id from the smile string query"""
        # molecule = new_client.molecule
        # ids =  molecule.filter(molecule_structures__canononical_smiles=).only(['molecule_chembl_id', ])
        # mol
        return [''] * len(mols)

    @staticmethod
    def get_smiles(mols):
        return [Chem.MolToSmiles(mol) for mol in mols]



class ProcessPDB():

    @classmethod
    def pdb_to_lmdb(cls, pdbs, residues_in_pocket_df, outlmdb_name):
        """Define pocket, Get pocket id str, pocket atom elements, and pocket_atom_coords, output to lmdb
        This function should only have inputted all the structures for a single pocket target. 
        pdbs: list of pdb filepaths. 
        out_lmdb_name
        
        """

        #unpack pocket def
        

        pocket_dicts = []
        for pdb in pdbs:
            # get pocket string?
            u = mda.Universe(pdb)
            pocket_residues = cls._get_pocket_from_centroid(pdb, (-62, -62, 17))
            pocket_names, pocket_coords = cls.get_pocket(u, pocket_residues)

            pocket_dict = {
                'pocket':cls.get_pocket_id(pdb),
                'pocket_atoms':pocket_names,
                'pocket_coordinates': pocket_coords,
            }
            pocket_dicts.append(pocket_dict)
            print(pocket_dict['pocket'], len(pocket_dict['pocket_atoms']))

        with lmdb.open(outlmdb_name, map_async=True) as opened_lmdb:
            for i, pocket_dict in enumerate(pocket_dicts):
                write_to_lmdb(opened_lmdb, i, pocket_dict)

    @staticmethod
    def get_pocket_id(pdb_fpath):
        return Path(pdb_fpath).name.rsplit(maxsplit=1)[0]


    @staticmethod
    def get_pocket(u, residues_in_pocket):
        """Returns the mda atom_group of the all residues deemed to be in the pocket. There are several ways a pocket can be / have been defined:
        1. Fpokcet to define pockets "10A in size" 
        2. Genpocket: residues with at least one heavy atom within 6A of the AI generated ligands
        3. Fpocket Centroid 
        4. Based on existing resolved ligand

        POCKET DETECTION MAY NEED TO OCCUR EARLIER. THIS FUNC SHOULD JUST READ SPECIFIED POCKET INFO AND RETURN ATOM PROPERTIES

        The pocket will be defined according the available information.
        If a resolved ligand exists for this structure, the pocket is all heavy atoms of residues with at least 1 heavy atom within 6A of the ligand,
        else, Genpocket is ran on the structure [NOT IMPLEMENTED YET].

        Or, while I sort my shit out, the pocketvec centroid is used, and all residue heavy atoms within 10A 

        """

        # read some pockect definition df
        # get a list of residue indices that can be passed to mdaanalysis

        selstr = " ".join([str(i) for i in residues_in_pocket])
        pocket_atoms = u.select_atoms(f'resindex {selstr}')

        return list(pocket_atoms.atoms.names), [coords_xyz for coords_xyz in pocket_atoms.positions]


    @staticmethod
    def _get_pocket_from_centroid(pdb, centroid, within=10):
        """This is a function to define a pocket residues on the basis of an inputted centroid.
        It's purpose was for testing class function before I got genpocket working. Should be unnecessary after that. 
        """

        u = mda.Universe(pdb)
        pocket_atoms = u.select_atoms(f'protein and (not type hydrogen) and point {centroid[0]} {centroid[1]} {centroid[2]} {within}')
        return pocket_atoms.atoms.residues.resindices



if __name__ == '__main__':

    import sys
    input_file = Path(sys.argv[1])

    if input_file.suffix == '.sdf':
        ProcessLigands.sdf_to_lmdb(input_file, 'test_sdf_lmdb') #The sdf should contain multiple ligands, but theoretically it could be modified to take multiple

    elif input_file.suffix == '.pdb':
        ProcessPDB.pdb_to_lmdb(sys.argv[1:], [], 'test_lmdb') # Will need to be updated to take in residues_in_pocket info from cli. 

    else:
        raise Exception(f'The inputted file {input_file} was not recognised! It should be either .sdf or .pdb')
    
    # print(input_file)
    # content = explore_lmdb(str(input_file))
    # print('original: ', len(content))
    # dict_to_lmdb(content)
    retrieved_content = explore_lmdb('test_sdf_lmdb/data.mdb')
    print('retireved: ', len(retrieved_content))
    # print(retrieved_content)

# To do
# Check no hydrogens are outputted
# Cleanup produced dir, rename the data.mdb, rm lock.mdb