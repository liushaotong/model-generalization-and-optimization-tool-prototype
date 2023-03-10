
from xgboost import XGBRegressor

def get_generalization(data):
    selectedTask = data['selectedTask']
    margin = data['1/margin']
    frobenius_initial = data['frobenius_initial']
    sum_frobenius = data['sum_frobenius']
    sum_spectral = data['sum_spectral']
    sum_spectral_margin = data['sum_spectral/margin']
    model = XGBRegressor()
    if selectedTask == 'cifar-10':
        # model.load_model()
        model.fit([[margin, frobenius_initial, sum_frobenius, sum_spectral, sum_spectral_margin]], [1])
        prediction = model.predict([[margin, frobenius_initial, sum_frobenius, sum_spectral, sum_spectral_margin]])
        value = {'泛化差距估计': float(prediction[0])}
        return value

    else:
        if selectedTask == 'svhn':
            # model.load_model()
            prediction = model.predict([[margin, frobenius_initial, sum_frobenius, sum_spectral, sum_spectral_margin]])
            value = {'泛化差距估计': float(prediction[0])}
            return value



if __name__ == '__main__':
    data = {}
    data['selectedTask'] = 'cifar-10'
    data['1/margin'] = 1
    data['frobenius_initial'] = 1
    data['sum_frobenius'] = 1
    data['sum_spectral'] = 1
    data['sum_spectral/margin'] = 1
    value = get_generalization(data)
    print(value)
