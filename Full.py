pip install tensorflow numpy pandas streamlit streamlit_star_rating
import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_star_rating import st_star_rating
st.title('MovieMate')
def reconstructed_output(h0_state, W, vb):
    v1_prob = tf.nn.sigmoid(tf.matmul(h0_state, tf.transpose(W)) + vb) 
    v1_state = tf.nn.relu(tf.sign(v1_prob - tf.random.uniform(tf.shape(v1_prob)))) 
    return v1_state[0]
def hidden_layer(v0_state, W, hb):
    h0_prob = tf.nn.sigmoid(tf.matmul([v0_state], W) + hb)  
    h0_state = tf.nn.relu(tf.sign(h0_prob - tf.random.uniform(tf.shape(h0_prob)))) 
    return h0_state
def error(v0_state, v1_state):
    return tf.reduce_mean(tf.square(v0_state - v1_state))
movies_df = pd.read_csv('movies.dat', sep='::', header=None, engine='python')
ratings_df = pd.read_csv('ratings.dat', sep='::', header=None, engine='python')
movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings_df.drop("Timestamp",axis=1,inplace=True)
userid=ratings_df["UserID"].iloc[-1]+1
op=movies_df['Title'].tolist()
k=st.selectbox("Select a Movie:",op)
stars1 = int(st_star_rating("Please rate your 1st experience", maxValue=5, defaultValue=0, key="rating1"))
k1=st.selectbox("Select second Movie:",op)
stars2 = int(st_star_rating("Please rate your 2nd experience", maxValue=5, defaultValue=0, key="rating2"))
k2=st.selectbox("Select third Movie:",op)
stars3 = int(st_star_rating("Please rate your 3rd experience", maxValue=5, defaultValue=0, key="rating3"))
if st.button("Recommend me!"):
    id1=movies_df.loc[movies_df["Title"]==k,"MovieID"].iloc[0]
    id2=movies_df.loc[movies_df["Title"]==k1,"MovieID"].iloc[0]
    id3=movies_df.loc[movies_df["Title"]==k2,"MovieID"].iloc[0]
    newr=pd.DataFrame({"UserID":[userid,userid,userid],"MovieID":[id1,id2,id3],"Rating":[stars1,stars2,stars3]})
    ratings_df=pd.concat([ratings_df,newr])
    user_rating_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')
    norm_user_rating_df = user_rating_df.fillna(0) / 5.0
    trX = norm_user_rating_df.values
    hiddenUnits = 20
    visibleUnits =  len(user_rating_df.columns)
    vb = tf.Variable(tf.zeros([visibleUnits]), tf.float32)
    hb = tf.Variable(tf.zeros([hiddenUnits]), tf.float32)
    W = tf.Variable(tf.zeros([visibleUnits, hiddenUnits]), tf.float32)
    v0 = tf.zeros([visibleUnits], tf.float32)
    tf.matmul([v0], W)
    h0 = hidden_layer(v0, W, hb)
    v1 = reconstructed_output(h0, W, vb)
    err = tf.reduce_mean(tf.square(v0 - v1))
    epochs = 1
    batchsize = 500
    errors = []
    weights = []
    K=1
    alpha = 0.1
    train_ds = \
        tf.data.Dataset.from_tensor_slices((np.float32(trX))).batch(batchsize)
    v0_state=v0
    for epoch in range(epochs):
        batch_number = 0
        for batch_x in train_ds:
            for i_sample in range(len(batch_x)):           
                for k in range(K):
                    v0_state = batch_x[i_sample]
                    h0_state = hidden_layer(v0_state, W, hb)
                    v1_state = reconstructed_output(h0_state, W, vb)
                    h1_state = hidden_layer(v1_state, W, hb)
                    delta_W = tf.matmul(tf.transpose([v0_state]), h0_state) - tf.matmul(tf.transpose([v1_state]), h1_state)
                    W = W + alpha * delta_W
                    vb = vb + alpha * tf.reduce_mean(v0_state - v1_state, 0)
                    hb = hb + alpha * tf.reduce_mean(h0_state - h1_state, 0) 
                    v0_state = v1_state
                if i_sample == len(batch_x)-1:
                    err = error(batch_x[i_sample], v1_state)
                    errors.append(err)
                    weights.append(W)
            batch_number += 1
    mock_user_id = userid
    inputUser = trX[mock_user_id-1].reshape(1, -1)
    inputUser = tf.convert_to_tensor(trX[mock_user_id-1],"float32")
    v0 = inputUser
    v0test = tf.zeros([visibleUnits], tf.float32)
    hh0 = tf.nn.sigmoid(tf.matmul([v0], W) + hb)
    vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
    rec = vv1
    scored_movies_df_mock = movies_df[movies_df['MovieID'].isin(user_rating_df.columns)]
    scored_movies_df_mock = scored_movies_df_mock.assign(RecommendationScore = rec[0])
    st.write("For you!")
    st.table(scored_movies_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20))
