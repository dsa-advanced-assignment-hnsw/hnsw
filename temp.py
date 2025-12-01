import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Khai báo Tên tệp và Chiều dữ liệu (Kích thước Nhúng)
file_configs = {
    # Nhóm 512 chiều
    'Model_A_512': {'filename': '.cache/clip-vit-base-patch32/image_embeddings/clip-vit-base-patch32_Images_Embedded_1_to_100000.h5', 'dim': 512, 'group': '512D'},
    'Model_B_512': {'filename': '.cache/clip-vit-base-patch16/image_embeddings/clip-vit-base-patch16_Images_Embedded_1_to_100000.h5', 'dim': 512, 'group': '512D'},
    # Nhóm 768 chiều
    'Model_C_768': {'filename': '.cache/clip-vit-large-patch14/image_embeddings/clip-vit-large-patch14_Images_Embedded_1_to_100000.h5', 'dim': 768, 'group': '768D'},
    'Model_D_768': {'filename': '.cache/clip-vit-large-patch14-336/image_embeddings/clip-vit-large-patch14-336_Images_Embedded_1_to_10000.h5', 'dim': 768, 'group': '768D'},
}

EMBEDDING_KEY = 'embeddings'
SAMPLE_SIZE = 10000

def load_and_sample_embeddings(filename, model_name, expected_dim, sample_size):
    """Tải và lấy mẫu (sampling) các embeddings từ tệp H5."""
    try:
        with h5py.File(filename, 'r') as f:
            embeddings = f[EMBEDDING_KEY][:]

            if embeddings.shape[1] != expected_dim:
                print(f"Lỗi kích thước: {model_name} có chiều {embeddings.shape[1]}, không khớp {expected_dim}. Bỏ qua.")
                return None

            if embeddings.shape[0] > sample_size:
                print(f"[{model_name}]: Lấy mẫu ngẫu nhiên {sample_size} điểm.")
                indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
                embeddings = embeddings[indices]

            return embeddings
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp {filename}")
        return None
    except KeyError:
        print(f"Lỗi: Không tìm thấy trường '{EMBEDDING_KEY}' trong tệp {filename}")
        return None

# 2. Tải Dữ liệu và Phân nhóm
data_by_group = {'512D': [], '768D': []}
stats = {}

for model_name, config in file_configs.items():
    embeddings = load_and_sample_embeddings(
        config['filename'],
        model_name,
        config['dim'],
        SAMPLE_SIZE
    )

    if embeddings is not None:
        data_by_group[config['group']].append(
            pd.DataFrame({
                'embedding': list(embeddings),
                'model': model_name
            })
        )

        stats[model_name] = {
            'Count': embeddings.shape[0],
            'Embedding_Dim': embeddings.shape[1],
            'Mean_Norm': np.mean(np.linalg.norm(embeddings, axis=1)),
            'Mean': np.mean(embeddings),
            'StdDev': np.std(embeddings),
        }

print("\n-------------------------------------------")
print("BẢNG SO SÁNH THỐNG KÊ CƠ BẢN CỦA EMBEDDINGS")
print("-------------------------------------------")
stats_df = pd.DataFrame(stats).T
print(stats_df)
print("-------------------------------------------\n")


# 3. Giảm chiều và Trực quan hóa Bằng PCA (Theo Nhóm Chiều)
for group_name, data_list in data_by_group.items():
    if not data_list:
        print(f"Không có dữ liệu hợp lệ cho nhóm {group_name}.")
        continue

    combined_df = pd.concat(data_list, ignore_index=True)
    X = np.array(combined_df['embedding'].tolist())

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Thực hiện PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['model'] = combined_df['model']

    # Trực quan hóa
    plt.figure(figsize=(10, 8))
    for name, group in pca_df.groupby('model'):
        plt.scatter(group['PC1'], group['PC2'], label=name, alpha=0.6, s=15)

    total_variance = pca.explained_variance_ratio_.sum() * 100
    plt.title(f'So sánh không gian nhúng bằng PCA ({group_name}) - Tổng phương sai: {total_variance:.2f}%')
    plt.xlabel(f'Thành phần chính 1 (Giải thích: {pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Thành phần chính 2 (Giải thích: {pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'pca_comparison_{group_name}.png')
    print(f"Đã lưu biểu đồ so sánh PCA cho nhóm {group_name} tại 'pca_comparison_{group_name}.png'")