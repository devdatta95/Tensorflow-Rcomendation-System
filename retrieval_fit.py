from retrieval_model import *

model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=EPOCHS)

model.evaluate(cached_test, return_dict=True)
# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model")

    # Save the index.
    tf.saved_model.save(index, path)

    # Load it back; can also be done in TensorFlow Serving.
    loaded = tf.saved_model.load(path)

    # Pass a user id in, get top predicted movie titles back.
    scores, titles = loaded(["42"])

    print(f"Recommendations: {titles[0][:3]}")

