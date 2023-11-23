from dt.main import DecisionTreeCV
from one_r.main import OneRule
from seqcov.main_cv import SeqCovCV
from rf.main import RandomForestCV
from ml_p.main_gs import MLPCV

folder, file, file_norm, file_cat = './report/SEV', '/sev.csv', '/sev_norm.csv', '/sev_cat.csv'

print('Decision tree running...')
dt = DecisionTreeCV(folder, file)
dt.run()

print('OneRule running...')
oner = OneRule(folder, file_cat)
oner.run()

print('Sequential covering running...')
sc = SeqCovCV(folder, file)
sc.run()

print('Random forest running...')
ranf = RandomForestCV(folder, file)
ranf.run()

print('MLP running...')
mlp = MLPCV(folder, file_norm)
mlp.run()