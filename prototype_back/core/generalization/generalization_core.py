
from xgboost import XGBRegressor

def get_generalization(data):
    selectedTask = data['selectedTask']
    margin = data['1/margin：']
    frobenius_initial = data['frobenius_initial：']
    sum_frobenius = data['sum_frobenius：']
    sum_spectral = data['sum_spectral：']
    sum_spectral_margin = data['sum_spectral/margin：']
    pacbayes_initial = data['pacbayes_initial：']
    pacbayes_origin = data['pacbayes_origin：']
    sharpness = data['sharpness：']
    model = XGBRegressor()
    if selectedTask == 'cifar-10':
        model.load_model('./core/generalization/model/xgbr_cifar10.json')
        # model.fit([[margin, frobenius_initial, sum_frobenius, sum_spectral, sum_spectral_margin]], [1])
        # prediction = model.predict([[margin, frobenius_initial, sum_frobenius, sum_spectral, sum_spectral_margin]])
        prediction = model.predict([[frobenius_initial, sum_frobenius, sum_spectral, sharpness, pacbayes_initial,
                                     pacbayes_origin, margin, sum_spectral_margin]])
        value = {'泛化差距估计：': float(prediction[0])}
        return value

    else:
        if selectedTask == 'svhn':
            model.load_model('./core/generalization/model/xgbr_svhn.json')
            prediction = model.predict(
                [[frobenius_initial, sum_frobenius, sum_spectral, sharpness, pacbayes_initial, pacbayes_origin,
                  margin, sum_spectral_margin]])
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
