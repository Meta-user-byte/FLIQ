from DIAHelper import Helper as DIAHelper

DIA_Helper = DIAHelper("Datasets/drug+induced+autoimmunity+prediction/DIA_trainingset_RDKit_descriptors.csv", 
                       "Datasets/drug+induced+autoimmunity+prediction/DIA_testset_RDKit_descriptors.csv")

print(DIA_Helper.get_col_idx_map())
print(DIA_Helper.get_col_prop_map(0))