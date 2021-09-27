from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files('rajyellow46/wine-quality', path='D:/Netology/Python/pyda-16-em/DS Project Flow/WineQualityPrediction/data_external/')