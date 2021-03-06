install:
	./venv/bin/python setup.py develop

naive_bayes:
	./venv/bin/python -m 'naive_bayes.nb_author_id'

terrain:
	./venv/bin/python -m 'terrain.student_main'

svm:
	./venv/bin/python -m 'svm.svm_author_id'

dt_exercise:
	./venv/bin/python -m 'decision_tree_exercise.student_main'

decision_tree:
	./venv/bin/python -m 'decision_tree.dt_author_id'

choose_you_own:
	./venv/bin/python -m 'choose_your_own.your_algorithm'

explore:
	./venv/bin/python -m 'datasets_questions.explore_enron_data'

regression:
	./venv/bin/python -m 'regression.finance_regression'

outliers:
	./venv/bin/python -m 'outliers.outlier_removal_regression'

enron_outliers:
	./venv/bin/python -m 'outliers.enron_outliers'

kmeans:
	./venv/bin/python -m 'k_means.k_means_cluster'

scaling:
	./venv/bin/python -m 'scaling'

parse_out_email_text:
	./venv/bin/python -m 'tools.parse_out_email_text'

vectorize_text:
	./venv/bin/python -m 'text_learning.vectorize_text'

find_signature:
	./venv/bin/python -m 'feature_selection.find_signature'

pca:
	./venv/bin/python -m 'pca.eigenfaces'

validation:
	./venv/bin/python -m 'validation.validate_poi'

evaluation:
	./venv/bin/python -m 'evaluation.evaluate_poi_identifier'
