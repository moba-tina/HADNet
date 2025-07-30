import numpy as np
from hmmlearn import hmm

def generate_counterfactual_data_and_save(input_data, model=None, num_samples=300, n_components=2):
    """
    生成反事实数据并保存到文件

    参数：
    - input_data: 输入的数据，形状为 (num_samples, num_features)
    - output_file_path: 保存生成数据的文件路径
    - model: 训练好的 HMM 模型，默认为 None，如果为 None，则在函数内部重新训练模型
    - num_samples: 生成的观测数据的数量，默认为 100
    - n_components: 隐含状态的数量，默认为 2

    返回：
    - generated_counterfactual_data: 生成的反事实数据
    """
    # 将输入数据转换为 HMM 能处理的格式
    observations = input_data.reshape(-1, input_data.shape[-1])

    # 定义和训练 HMM 模型
    if model is None:
        # model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="spherical", n_iter=100)

        model.fit(observations)

    # 生成反事实数据
    # new_hidden_states = np.random.randint(model.n_components, size=num_samples)
    generated_counterfactual_data, _ = model.sample(num_samples, random_state=np.random.RandomState(42))

    # 保存输入输出数据到文件
    # np.savetxt(output_file_path, generated_counterfactual_data, delimiter=",", fmt="%.6f")

    # print("生成的反事实数据已保存到文件:", output_file_path)

    return generated_counterfactual_data

# 示例使用
if __name__ == "__main__":
    # 生成示例数据集
    np.random.seed(42)
    data = np.random.randn(100, 128)  # 假设您有实际的异常特征数据

    # 生成反事实数据并保存到文件
    generated_counterfactual_data = generate_counterfactual_data_and_save(data,num_samples=300)
    # print(generated_counterfactual_data.shape)
