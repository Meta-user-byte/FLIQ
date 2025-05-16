import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Helper:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

        self.len_train_df = -1
        self.len_test_df = -1

        self.train_df = self.load_train_dataset()
        self.test_df = self.load_test_dataset()

        self.col_idx_map = self.get_col_idx_map()

        self.col_prop_map = {
            # General Identifiers
            'Label': 'Binary indicator denoting whether a compound is associated with drug-induced autoimmunity (1 for positive, 0 for negative).',
            'SMILES': 'Simplified Molecular Input Line Entry System; a textual representation of a molecule\'s structure.',

            # Topological and Connectivity Descriptors
            'BalabanJ': 'A topological index reflecting molecular connectivity and complexity.',
            'BertzCT': 'A complexity index based on molecular graph theory.',
            'Chi0': 'Zero-order molecular connectivity index; considers atom valence.',
            'Chi0n': 'Zero-order molecular connectivity index for non-hydrogen atoms.',
            'Chi0v': 'Zero-order valence molecular connectivity index.',
            'Chi1': 'First-order molecular connectivity index; considers pairs of bonded atoms.',
            'Chi1n': 'First-order molecular connectivity index for non-hydrogen atoms.',
            'Chi1v': 'First-order valence molecular connectivity index.',
            'Chi2n': 'Second-order molecular connectivity index for non-hydrogen atoms.',
            'Chi2v': 'Second-order valence molecular connectivity index.',
            'Chi3n': 'Third-order molecular connectivity index for non-hydrogen atoms.',
            'Chi3v': 'Third-order valence molecular connectivity index.',
            'Chi4n': 'Fourth-order molecular connectivity index for non-hydrogen atoms.',
            'Chi4v': 'Fourth-order valence molecular connectivity index.',

            # Electrotopological State Descriptors
            'EState_VSA1': 'Sum of electrotopological state values over specific van der Waals surface area range 1.',
            'EState_VSA2': 'Sum over range 2.',
            'EState_VSA3': 'Sum over range 3.',
            'EState_VSA4': 'Sum over range 4.',
            'EState_VSA5': 'Sum over range 5.',
            'EState_VSA6': 'Sum over range 6.',
            'EState_VSA7': 'Sum over range 7.',
            'EState_VSA8': 'Sum over range 8.',
            'EState_VSA9': 'Sum over range 9.',
            'EState_VSA10': 'Sum over range 10.',
            'EState_VSA11': 'Sum over range 11.',
            'MaxAbsEStateIndex': 'Maximum absolute electrotopological state value in the molecule.',
            'MaxEStateIndex': 'Maximum electrotopological state value in the molecule.',
            'MinAbsEStateIndex': 'Minimum absolute electrotopological state value in the molecule.',
            'MinEStateIndex': 'Minimum electrotopological state value in the molecule.',

            # Partial Charge Descriptors
            'MaxAbsPartialCharge': 'Maximum absolute partial atomic charge in the molecule.',
            'MaxPartialCharge': 'Maximum partial atomic charge in the molecule.',
            'MinAbsPartialCharge': 'Minimum absolute partial atomic charge in the molecule.',
            'MinPartialCharge': 'Minimum partial atomic charge in the molecule.',

            # Molecular Properties
            'ExactMolWt': 'Exact molecular weight calculated using the exact isotopic masses.',
            'MolWt': 'Average molecular weight based on standard atomic weights.',
            'MolLogP': 'Logarithm of the partition coefficient between octanol and water; indicates hydrophobicity.',
            'MolMR': 'Molar refractivity; related to the molecule\'s polarizability.',
            'FractionCSP3': 'Fraction of sp³-hybridized carbon atoms; indicates saturation level.',
            'LabuteASA': 'Approximate surface area calculated using Labute\'s method.',
            'TPSA': 'Topological Polar Surface Area; sum of surface areas of polar atoms.',

            # Atom and Bond Counts
            'HeavyAtomCount': 'Number of non-hydrogen atoms.',
            'HeavyAtomMolWt': 'Molecular weight of heavy atoms.',
            'NumHAcceptors': 'Number of hydrogen bond acceptors.',
            'NumHDonors': 'Number of hydrogen bond donors.',
            'NumHeteroatoms': 'Number of atoms other than carbon and hydrogen.',
            'NumRadicalElectrons': 'Number of unpaired electrons.',
            'NumRotatableBonds': 'Number of bonds that allow free rotation.',
            'NumValenceElectrons': 'Total number of valence electrons.',

            # Ring and Cycle Descriptors
            'RingCount': 'Total number of rings in the molecule.',
            'NumAliphaticCarbocycles': 'Number of aliphatic carbocyclic rings.',
            'NumAliphaticHeterocycles': 'Number of aliphatic heterocyclic rings.',
            'NumAliphaticRings': 'Total number of aliphatic rings.',
            'NumAromaticCarbocycles': 'Number of aromatic carbocyclic rings.',
            'NumAromaticHeterocycles': 'Number of aromatic heterocyclic rings.',
            'NumAromaticRings': 'Total number of aromatic rings.',
            'NumSaturatedCarbocycles': 'Number of saturated carbocyclic rings.',
            'NumSaturatedHeterocycles': 'Number of saturated heterocyclic rings.',
            'NumSaturatedRings': 'Total number of saturated rings.',

            # Kappa Shape Indices
            'Kappa1': 'First-order kappa shape index; indicates molecular flexibility.',
            'Kappa2': 'Second-order kappa shape index; relates to molecular shape.',
            'Kappa3': 'Third-order kappa shape index; provides information on molecular branching.',

            # Information Content and Complexity
            'Ipc': 'Information content index; measures molecular complexity.',
            'HallKierAlpha': 'Descriptor related to molecular size and branching.',

            # Surface Area Descriptors
            'PEOE_VSA1': 'Sum of partial charges over specific van der Waals surface area range 1.',
            'PEOE_VSA2': 'Sum over range 2.',
            'PEOE_VSA3': 'Sum over range 3.',
            'PEOE_VSA4': 'Sum over range 4.',
            'PEOE_VSA5': 'Sum over range 5.',
            'PEOE_VSA6': 'Sum over range 6.',
            'PEOE_VSA7': 'Sum over range 7.',
            'PEOE_VSA8': 'Sum over range 8.',
            'PEOE_VSA9': 'Sum over range 9.',
            'PEOE_VSA10': 'Sum over range 10.',
            'PEOE_VSA11': 'Sum over range 11.',
            'PEOE_VSA12': 'Sum over range 12.',
            'PEOE_VSA13': 'Sum over range 13.',
            'PEOE_VSA14': 'Sum over range 14.',

            'SMR_VSA1': 'Sum of molar refractivity over specific van der Waals surface area range 1.',
            'SMR_VSA2': 'Sum over range 2.',
            'SMR_VSA3': 'Sum over range 3.',
            'SMR_VSA4': 'Sum over range 4.',
            'SMR_VSA5': 'Sum over range 5.',
            'SMR_VSA6': 'Sum over range 6.',
            'SMR_VSA7': 'Sum over range 7.',
            'SMR_VSA8': 'Sum over range 8.',
            'SMR_VSA9': 'Sum over range 9.',
            'SMR_VSA10': 'Sum over range 10.',

            'SlogP_VSA1': 'Sum of logP contributions over specific van der Waals surface area range 1.',
            'SlogP_VSA2': 'Sum over range 2.',
            'SlogP_VSA3': 'Sum over range 3.',
            'SlogP_VSA4': 'Sum over range 4.',
            'SlogP_VSA5': 'Sum over range 5.',
            'SlogP_VSA6': 'Sum over range 6.',
            'SlogP_VSA7': 'Sum over range 7.',
            'SlogP_VSA8': 'Sum over range 8.',
            'SlogP_VSA9': 'Sum over range 9.',
            'SlogP_VSA10': 'Sum over range 10.',
            'SlogP_VSA11': 'Sum over range 11.',
            'SlogP_VSA12': 'Sum over range 12.',

            'VSA_EState1': 'Sum of electrotopological state values over specific van der Waals surface area range 1.',
            'VSA_EState2': 'Sum over range 2.',
            'VSA_EState3': 'Sum over range 3.',
            'VSA_EState4': 'Sum over range 4.',
            'VSA_EState5': 'Sum over range 5.',
            'VSA_EState6': 'Sum over range 6.',
            'VSA_EState7': 'Sum over range 7.',
            'VSA_EState8': 'Sum over range 8.',
            'VSA_EState9': 'Sum over range 9.',
            'VSA_EState10': 'Sum over range 10.',

            # Functional Group Counts (fr_ prefixed)
            'fr_Al_COO': 'Number of aliphatic carboxylic acid groups.',
            'fr_Al_OH': 'Number of aliphatic hydroxyl groups.',
            'fr_Al_OH_noTert': 'Number of aliphatic hydroxyl groups excluding tertiary alcohols.',
            'fr_ArN': 'Number of nitrogen functional groups attached to aromatic systems.',
            'fr_Ar_COO': 'Number of aromatic carboxylic acid groups.',
            'fr_Ar_N': 'Number of aromatic nitrogen atoms.',
            'fr_Ar_NH': 'Number of aromatic amine groups.',
            'fr_Ar_OH': 'Number of aromatic hydroxyl groups (phenols).',
            'fr_COO': 'Number of carboxylic acid groups.',
            'fr_COO2': 'Number of carboxylic acid groups.',
            'fr_C_O': 'Number of carbonyl oxygen atoms.',
            'fr_C_O_noCOO': 'Number of carbonyl oxygen atoms excluding those in carboxylic acids.',
            'fr_C_S': 'Number of thiocarbonyl groups',
            'fr_HOCCN': "Presence of HO–C–C–N motif (hydroxyethylamine or similar)",
            'fr_Imine': "Imine group (C=NH or C=NR)",
            'fr_NH0': "Tertiary amines or quaternary N (no attached hydrogen)",
            'fr_NH1': "Secondary amines (one hydrogen on nitrogen)",
            'fr_NH2': "Primary amines (two hydrogens on nitrogen)",
            'fr_N_O': "Nitroso group (N–O)",
            'fr_Ndealkylation1': "Likely site of mono-N-dealkylation",
            'fr_Ndealkylation2': "Likely site of bis-N-dealkylation",
            'fr_Nhpyrrole': "Pyrrole nitrogen (in 5-membered heterocycles)",
            'fr_SH': "Thiol group (–SH)",
            'fr_aldehyde': "Aldehyde group (–CHO)",
            'fr_alkyl_carbamate': "Alkyl carbamate (R–O–C(=O)–NR2)",
            'fr_alkyl_halide': "Alkyl halides (R–X, where X = F, Cl, Br, I)",
            'fr_allylic_oxid': "Allylic alcohol or oxidation site (C=C–C–OH)",
            'fr_amide': "Amide group (R–CO–NR2)",
            'fr_amidine': "Amidines (C(=NH)–NH2 or variants)",
            'fr_aniline': "Aniline structure (aromatic amine)",
            'fr_aryl_methyl': "Aromatic methyl groups (Ar–CH3)",
            'fr_azide': "Azide group (–N₃)",
            'fr_azo': "Azo group (R–N=N–R')",
            'fr_barbitur': "Barbiturate-like structure",
            'fr_benzene': "Benzene ring present",
            'fr_benzodiazepine': "Benzodiazepine scaffold",
            'fr_bicyclic': "Fused or bridged bicyclic ring systems",
            'fr_diazo': "Diazo group (–N=N+)",
            'fr_dihydropyridine': "Dihydropyridine ring (partially reduced pyridine)",
            'fr_epoxide': "Epoxide ring (three-membered ether)",
            'fr_ester': "Ester group (R–COO–R')",
            'fr_ether': "Ether group (R–O–R')",
            'fr_furan': "Furan ring (5-membered oxygen heterocycle)",
            'fr_guanido': "Guanidine group (HNC(NH2)2)",
            'fr_halogen': "Any halogen atom (F, Cl, Br, I)",
            'fr_hdrzine': "Hydrazine group (–NH–NH2)",
            'fr_hdrzone': "Hydrazone group (C=NNH2)",
            'fr_imidazole': "Imidazole ring",
            'fr_imide': "Imide group (two acyl groups on same nitrogen)",
            'fr_isocyan': "Isocyanate group (–N=C=O)",
            'fr_isothiocyan': "Isothiocyanate group (–N=C=S)",
            'fr_ketone': "Ketone group (C=O between carbons)",
            'fr_ketone_Topliss': "Ketone in Topliss QSAR pattern",
            'fr_lactam': "Cyclic amide (lactam ring)",
            'fr_lactone': "Cyclic ester (lactone ring)",
            'fr_methoxy': "Methoxy group (–OCH₃)",
            'fr_morpholine': "Morpholine ring (O- and N-containing six-membered ring)",
            'fr_nitrile': "Nitrile group (–C≡N)",
            'fr_nitro': "Nitro group (–NO₂)",
            'fr_nitro_arom': "Aromatic nitro group (Ar–NO₂)",
            'fr_nitro_arom_nonortho': "Aromatic nitro not ortho-substituted",
            'fr_nitroso': "Nitroso group (–NO)",
            'fr_oxazole': "Oxazole ring (5-membered N, O heterocycle)",
            'fr_oxime': "Oxime group (C=NOH)",
            'fr_para_hydroxylation': "Para-hydroxylated aromatic ring",
            'fr_phenol': "Phenol group (Ar–OH)",
            'fr_phenol_noOrthoHbond': "Phenol without ortho-H-bonding",
            'fr_phos_acid': "Phosphoric acid group (P–OH)",
            'fr_phos_ester': "Phosphate ester group",
            'fr_piperdine': "Piperidine ring (saturated N-heterocycle)",
            'fr_piperzine': "Piperazine ring (N–C–C–N six-membered ring)",
            'fr_priamide': "Primary amide (–CONH₂)",
            'fr_prisulfonamd': "Primary sulfonamide (–SO₂NH₂)",
            'fr_pyridine': "Pyridine ring (aromatic nitrogen heterocycle)",
            'fr_quatN': "Quaternary ammonium ion (N⁺R₄)",
            'fr_sulfide': "Sulfide group (R–S–R')",
            'fr_sulfonamd': "Sulfonamide (R–SO₂–NR₂)",
            'fr_sulfone': "Sulfone group (R–SO₂–R')",
            'fr_term_acetylene': "Terminal alkyne group (–C≡CH)",
            'fr_tetrazole': "Tetrazole ring (5-membered N₄ heterocycle)",
            'fr_thiazole': "Thiazole ring (5-membered S, N heterocycle)",
            'fr_thiocyan': "Thiocyanate group (–SCN or:NCS)",
            'fr_thiophene': "Thiophene ring (5-membered S heterocycle)",
            'fr_unbrch_alkane': "Unbranched alkane chain",
            'fr_urea': "Urea group (–NH–CO–NH–)"}
 

    
    def load_train_dataset(self):
        try:
            train_df = pd.read_csv(self.train_path)
            self.len_train_df = train_df.shape[0]
            return train_df
        except Exception as e_train_read_error:
            print("Error reading train dataset:", e_train_read_error)
        return None
    def load_test_dataset(self):
        try:
            test_df = pd.read_csv(self.test_path)
            self.len_test_df = test_df.shape[0]
            return test_df
        except Exception as e_test_read_error:
            print("Error reading test dataset:", e_test_read_error)
        return None
    
    def return_train_dataset(self, cols: list[int]):
        return self.train_df
    def return_test_dataset(self):
        return self.test_df
    
    def __getitem__(self, idx: int, train: bool = True):
        if train:
            if self.train_df is None:
                raise ValueError("Train dataset is not loaded.")
            if idx >= self.len_train_df:
                raise IndexError("Index out of range for train dataset.")
            return self.train_df.iloc[idx]
        else:
            if self.test_df is None:
                raise ValueError("Test dataset is not loaded.")
            if idx >= self.len_test_df:
                raise IndexError("Index out of range for test dataset.")
            return self.test_df.iloc[idx]
    
    def get_train_dataset_length(self):
        if self.train_df is None:
            raise ValueError("Train dataset is not loaded.")
        return self.len_train_df
    def get_test_dataset_length(self):
        if self.test_df is None:
            raise ValueError("Test dataset is not loaded.")
        return self.len_test_df
    
    def get_col_idx_map(self):
        if self.train_df is None:
            if self.test_df is None:
                raise ValueError("Train and test datasets are not loaded.")
            
            test_columns = self.test_df.columns
            
            col_idx_map = {}
            for i, col in enumerate(test_columns):
                # col_idx_map[col] = i
                col_idx_map[i] = col
            
            return col_idx_map
        
        else:
            train_columns = self.train_df.columns
            
            col_idx_map = {}
            for i, col in enumerate(train_columns):
                # col_idx_map[col] = i
                col_idx_map[i] = col
            
            return col_idx_map
    
    def get_col_prop_map(self, col_idx: int = None):
        if (self.train_df is not None) and (self.test_df is not None):
            if col_idx is not None:
                return self.col_idx_map[col_idx] + " --> " + self.col_prop_map[self.col_idx_map[col_idx]]
            else:
                return self.col_prop_map