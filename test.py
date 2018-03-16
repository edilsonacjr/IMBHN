

from IMBHN_adj import IMBHN
import pandas as pd
from time import time

df = pd.read_csv('/home/edilson/Dropbox/IMBHN/Teste/line.pos_context_1.csv', sep=',')

y = df['classe']

df.drop('classe', axis=1, inplace=True)

X = df.as_matrix()
#X = df.to_sparse()

clf = IMBHN()



from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy import sparse
from sklearn.utils import shuffle

#clf = DecisionTreeClassifier(random_state=1)

#clf = SVC(kernel='linear', C=1, random_state=1)

# variar o random state de acordo com as iterações
X, y = shuffle(X, y, random_state=0)
#X = sparse.csr_matrix(X)


t0 = time()
skf = StratifiedKFold(y, 2, random_state=1)#, shuffle=True)
scores = cross_validation.cross_val_score(clf, X, y, cv=skf, n_jobs=3)
print('Score 10-fold stratified', scores.mean())
print("Training time:", round(time()-t0, 10))