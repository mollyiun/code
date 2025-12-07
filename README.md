# mf_recommender_app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
import time

st.set_page_config(page_title="Матрицалық факторизация — Ұсыныс жүйелері", layout="wide")

# ----------------------
# Utilities / MF model
# ----------------------
def generate_synthetic_data(n_users=50, n_items=40, n_factors=3, noise=0.3, density=0.6, random_state=42):
    rng = np.random.RandomState(random_state)
    U_true = rng.normal(scale=1.0, size=(n_users, n_factors))
    V_true = rng.normal(scale=1.0, size=(n_items, n_factors))
    R_full = U_true @ V_true.T
    # normalize to 1-5
    R_min, R_max = R_full.min(), R_full.max()
    R_scaled = 1 + 4 * (R_full - R_min) / (R_max - R_min)
    R_noisy = R_scaled + rng.normal(scale=noise, size=R_scaled.shape)
    # mask some entries
    mask = rng.rand(n_users, n_items) < density
    R = np.where(mask, np.clip(np.round(R_noisy, 1), 1.0, 5.0), np.nan)
    # return as DataFrame of (user,item,rating)
    rows, cols = np.where(~np.isnan(R))
    df = pd.DataFrame({
        "user_id": rows,
        "item_id": cols,
        "rating": R[rows, cols].astype(float)
    })
    return df, R, U_true, V_true

def create_matrix_from_df(df, n_users=None, n_items=None):
    if n_users is None:
        n_users = int(df['user_id'].max()) + 1
    if n_items is None:
        n_items = int(df['item_id'].max()) + 1
    R = np.full((n_users, n_items), np.nan)
    for _, row in df.iterrows():
        R[int(row['user_id']), int(row['item_id'])] = float(row['rating'])
    return R

def train_mf_sgd(R, n_factors=10, lr=0.01, reg=0.02, n_epochs=50, verbose=False, seed=0):
    """
    Simple matrix factorization with SGD and bias terms.
    R: 2D numpy array with np.nan for missing entries
    Returns: P (users x k), Q (items x k), bu, bi, loss_history
    """
    rng = np.random.RandomState(seed)
    n_users, n_items = R.shape
    # initialize
    k = n_factors
    P = 0.1 * rng.randn(n_users, k)
    Q = 0.1 * rng.randn(n_items, k)
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    # global mean
    mask = ~np.isnan(R)
    mu = np.nanmean(R)
    # precompute indices
    samples = [(i, j, R[i, j]) for i in range(n_users) for j in range(n_items) if not np.isnan(R[i, j])]
    loss_history = []
    for epoch in range(n_epochs):
        rng.shuffle(samples)
        total_loss = 0.0
        for (i, j, r) in samples:
            pred = mu + bu[i] + bi[j] + P[i, :].dot(Q[j, :])
            e = r - pred
            total_loss += e**2
            # gradients
            bu[i] += lr * (e - reg * bu[i])
            bi[j] += lr * (e - reg * bi[j])
            P[i, :] += lr * (e * Q[j, :] - reg * P[i, :])
            Q[j, :] += lr * (e * P[i, :] - reg * Q[j, :])
        rmse = np.sqrt(total_loss / len(samples)) if len(samples)>0 else 0.0
        loss_history.append(rmse)
        if verbose and (epoch % max(1, n_epochs // 10) == 0):
            st.write(f"Epoch {epoch+1}/{n_epochs} RMSE={rmse:.4f}")
    return P, Q, bu, bi, mu, loss_history

def predict_all(P, Q, bu, bi, mu):
    return mu + bu.reshape(-1,1) + bi.reshape(1,-1) + P.dot(Q.T)

def compute_rmse_on_mask(R_true, R_pred, mask):
    diff = R_true[mask] - R_pred[mask]
    return np.sqrt(np.nanmean(diff**2))

def top_n_recommendations(pred_matrix, user_id, known_mask, n=10, item_names=None):
    user_scores = pred_matrix[user_id]
    # exclude known items
    idxs = np.argsort(-user_scores)
    recommended = [i for i in idxs if not known_mask[user_id, i]]
    recommended = recommended[:n]
    if item_names is not None:
        return [(i, item_names[i], user_scores[i]) for i in recommended]
    else:
        return [(i, None, user_scores[i]) for i in recommended]

# ----------------------
# UI layout
# ----------------------
st.title("Ұсыныс жүйелерінде — Матрицалық факторизация (Matrix Factorization Demo)")
st.write("Интерактивті қосымша: матрицалық факторизацияны (MF) қолдану арқылы ұсыныстар жасау және модельдің жұмысын визуализациялау.")

# Sidebar: data options
st.sidebar.header("Деректер / Пайдаланушы опциялар")
data_source = st.sidebar.radio("Деректер қайдан алынады?", ["Синтетикалық деректер (жасанды)", "Загрузить CSV (user,item,rating)"])

if data_source == "Синтетикалық деректер (жасанды)":
    n_users = st.sidebar.slider("Пайдаланушылар саны (n_users)", 10, 500, 80)
    n_items = st.sidebar.slider("Элементтер саны (n_items)", 10, 500, 60)
    true_factors = st.sidebar.slider("Шығарушы (true) факторлар саны (симуляцияда)", 1, 10, 3)
    density = st.sidebar.slider("Қол жетімді бағалардың тығыздығы", 0.05, 1.0, 0.35)
    noise = st.sidebar.slider("Шуды қосу (noise)", 0.0, 1.0, 0.2)
    random_state = st.sidebar.number_input("Random seed", 0, 9999, 42)
    if st.sidebar.button("Генерациялау"):
        with st.spinner("Синтетикалық деректер генерациялануда..."):
            df_ratings, R_full, U_true, V_true = generate_synthetic_data(
                n_users=n_users, n_items=n_items, n_factors=true_factors, noise=noise, density=density, random_state=random_state
            )
        st.session_state['ratings_df'] = df_ratings
        st.session_state['R_matrix'] = create_matrix_from_df(df_ratings, n_users=n_users, n_items=n_items)
        st.success("Деректер жасалды және жүктелді.")
else:
    uploaded = st.sidebar.file_uploader("Загрузите CSV (user_id,item_id,rating)", type=["csv"])
    if uploaded is not None:
        df_ratings = pd.read_csv(uploaded)
        # ensure columns
        if not {'user_id','item_id','rating'}.issubset(df_ratings.columns):
            st.sidebar.error("CSV файлында колонкалар болуы тиіс: user_id,item_id,rating")
        else:
            st.session_state['ratings_df'] = df_ratings
            st.session_state['R_matrix'] = create_matrix_from_df(df_ratings)

# If no data yet — generate small default synthetic
if 'R_matrix' not in st.session_state:
    df_ratings, R_full, U_true, V_true = generate_synthetic_data(n_users=80, n_items=60, n_factors=3, noise=0.2, density=0.35, random_state=42)
    st.session_state['ratings_df'] = df_ratings
    st.session_state['R_matrix'] = create_matrix_from_df(df_ratings, n_users=80, n_items=60)

# Main columns
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Деректер (ratings)")
    st.dataframe(st.session_state['ratings_df'].head(200))

    # show rating density stats
    R = st.session_state['R_matrix']
    n_users, n_items = R.shape
    n_observed = np.sum(~np.isnan(R))
    st.markdown(f"- Пайдаланушылар: **{n_users}**  \n- Элементтер: **{n_items}**  \n- Бақылаулар: **{int(n_observed)}** ({(n_observed / (n_users*n_items))*100:.1f}% тығыздық)")

with col2:
    st.subheader("Матрица (қысқаша көрініс)")
    # show heatmap of observed ratings (small matrices only)
    if n_users <= 120 and n_items <= 120:
        heat_df = pd.DataFrame(np.where(np.isnan(R), np.nan, R))
        fig = px.imshow(heat_df.fillna(0), labels=dict(x="item_id", y="user_id", color="rating"),
                        title="Матрицаның көрінісі (NaN = 0)", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Матрицаны көрсету үшін өлшемдер тым үлкен (<=120 талап етіледі).")

st.markdown("---")

# Training controls
st.header("Матрицалық факторизация — Оқыту (Model training)")
col_a, col_b, col_c = st.columns(3)
with col_a:
    n_factors = st.slider("Latent факторлар саны (k)", 2, 200, 20)
    lr = st.number_input("Оқу жылдамдығы (learning rate)", min_value=1e-5, max_value=1.0, value=0.01, format="%.5f")
with col_b:
    reg = st.number_input("Регуляризация (λ)", min_value=0.0, max_value=1.0, value=0.02, format="%.4f")
    n_epochs = st.slider("Эпохалар саны", 1, 500, 60)
with col_c:
    test_size = st.slider("Test бөлігі (%)", 5, 50, 20)
    seed = st.number_input("Seed", 0, 9999, 42)

train_button = st.button("Оқыту (Train MF)")

if train_button:
    R = st.session_state['R_matrix']
    # make train/test split on observed entries
    rows, cols = np.where(~np.isnan(R))
    pairs = list(zip(rows.tolist(), cols.tolist()))
    # create list of (i,j,r)
    samples = [(i,j,R[i,j]) for i,j in pairs]
    df_samples = pd.DataFrame(samples, columns=["user","item","rating"])
    train_df, test_df = train_test_split(df_samples, test_size=test_size/100.0, random_state=seed)
    # create matrices
    R_train = np.full(R.shape, np.nan)
    R_test = np.full(R.shape, np.nan)
    for _, r in train_df.iterrows():
        R_train[int(r.user), int(r.item)] = float(r.rating)
    for _, r in test_df.iterrows():
        R_test[int(r.user), int(r.item)] = float(r.rating)
    with st.spinner("Модельді оқыту... (SGD)"):
        start = time.time()
        P, Q, bu, bi, mu, loss_history = train_mf_sgd(R_train, n_factors=n_factors, lr=lr, reg=reg, n_epochs=n_epochs, verbose=False, seed=seed)
        elapsed = time.time() - start
    st.success(f"Оқыту аяқталды. Уақыт: {elapsed:.2f}s")
    st.session_state['mf_model'] = {"P":P, "Q":Q, "bu":bu, "bi":bi, "mu":mu, "loss":loss_history, "R_train":R_train, "R_test":R_test}

# If model trained, show results
if 'mf_model' in st.session_state:
    model = st.session_state['mf_model']
    P, Q, bu, bi, mu = model['P'], model['Q'], model['bu'], model['bi'], model['mu']
    R_train, R_test = model['R_train'], model['R_test']
    pred_matrix = predict_all(P, Q, bu, bi, mu)
    st.subheader("Нәтижелер және бағалау")
    col1, col2 = st.columns([1,1])
    with col1:
        # loss chart
        loss = model['loss']
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=loss, mode="lines+markers", name="Train RMSE per epoch"))
        fig.update_layout(title="Оқыту барысы (RMSE per epoch)", xaxis_title="Epoch", yaxis_title="RMSE")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # compute RMSE on train/test
        mask_train = ~np.isnan(R_train)
        mask_test = ~np.isnan(R_test)
        rmse_train = compute_rmse_on_mask(R_train, pred_matrix, mask_train) if mask_train.any() else np.nan
        rmse_test = compute_rmse_on_mask(R_test, pred_matrix, mask_test) if mask_test.any() else np.nan
        st.metric("Train RMSE", f"{rmse_train:.4f}")
        st.metric("Test RMSE", f"{rmse_test:.4f}")
    st.markdown("**Салыстыру: нақты баға және модель болжамы (бірнеше жол)**")
    # show a few comparisons
    comp_rows = []
    for i,j in zip(*np.where(~np.isnan(R_test))):
        comp_rows.append({"user":int(i), "item":int(j), "rating_true": float(R_test[i,j]), "rating_pred": float(pred_matrix[i,j])})
        if len(comp_rows) >= 10:
            break
    if len(comp_rows) > 0:
        st.dataframe(pd.DataFrame(comp_rows))

    st.markdown("---")
    st.subheader("Ұсыныс жасау (Recommendations)")
    u_select = st.number_input("Пайдаланушы ID таңдау", min_value=0, max_value=n_users-1 if 'n_users' in locals() else pred_matrix.shape[0]-1, value=0)
    top_k = st.number_input("Ұсыныс саны (top N)", min_value=1, max_value=50, value=10)
    if st.button("Топ ұсыныстар көрсету"):
        known_mask = ~np.isnan(st.session_state['R_matrix'])
        recs = top_n_recommendations(pred_matrix, int(u_select), known_mask, n=int(top_k))
        rec_df = pd.DataFrame([{"item_id": r[0], "score": float(r[2])} for r in recs])
        st.table(rec_df)

    st.markdown("---")
    st.subheader("Латенттік векторларды визуализациялау (PCA арқылы)")
    # reduce P and Q to 2D
    pca = PCA(n_components=2)
    try:
        P2 = pca.fit_transform(P)
        Q2 = pca.fit_transform(Q)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=P2[:,0], y=P2[:,1], mode='markers', name='Users', marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=Q2[:,0], y=Q2[:,1], mode='markers', name='Items', marker=dict(size=8, symbol='diamond')))
        fig.update_layout(title="Латенттік факторлар: қолданушылар vs элементтер (PCA 2D)", xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("PCA визуализация кезінде қате: " + str(e))

    st.markdown("---")
    st.subheader("Матрица қайта қалпына келтірілген (Reconstructed matrix)")
    if n_users <= 120 and n_items <= 120:
        fig = px.imshow(pred_matrix, labels=dict(x="item_id", y="user_id", color="pred_rating"), aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Reconstructed matrix view disabled for big matrices (<=120 required).")

st.markdown("---")
st.write("Қосымша түсініктеме: бұл демонстрация матрицалық факторизацияның негізгі идеяларын көрсетуге арналған: берілген рейтинг матрицасын бөлу — қолданушы және элемент матрицаларына (латент векторлар) — және сол арқылы бағаларды қайта құрастыру арқылы ұсыныстар жасау.")
