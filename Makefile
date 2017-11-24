install:
	python setup.py develop

naive_bayes:
	python -m 'naive_bayes.nb_author_id'

terrain:
	python -m 'terrain.student_main'

svm:
	python -m 'svm.svm_author_id'

dt_exercise:
	python -m 'decision_tree_exercise.student_main'

decision_tree:
	python -m 'decision_tree.dt_author_id'

choose_you_own:
	python -m 'choose_your_own.your_algorithm'

explore:
	./venv/bin/python -m 'datasets_questions.explore_enron_data'
