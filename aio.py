from dt.main import DecisionTreeCV
from one_r.main import OneRule
from seqcov.main_cv import SeqCovCV
from rf.main import RandomForestCV
from ml_p.main_gs import MLPCV

# Evaluates all algorithms - GS+CV to get the best parameters
# and scoring on test set using the calculated parameters (except OneR)

folder, file, file_norm, file_cat = './report/SEV', '/data.csv', '/data_norm.csv', '/data_cat.csv'

# print('Decision tree running...')
# dt = DecisionTreeCV(folder, file)
# dt.run()

# print('OneRule running...')
# oner = OneRule(folder, file_cat)
# oner.run()

# print('Sequential covering running...')
# sc = SeqCovCV(folder, file)
# sc.run()

# print('Random forest running...')
# ranf = RandomForestCV(folder, file)
# ranf.run()

print('MLP running...')
mlp = MLPCV(folder, file_norm)
mlp.run()