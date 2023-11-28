from autogluon.tabular import TabularDataset, TabularPredictor

data_folder_path = '../'
train_data = TabularDataset(data_folder_path + 'all_data.csv')
test_data = TabularDataset(data_folder_path + 'all_data.csv')

predictor = TabularPredictor(label='Label').fit(train_data=train_data,
                                                ag_args_fit={"ag.max_memory_usage_ratio": 0.7})
predictions = predictor.predict(test_data)
