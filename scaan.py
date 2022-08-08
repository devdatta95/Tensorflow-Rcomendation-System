
from retrieval_fit import *
from retrieval_stage import *


# by using
scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
scann_index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

# Get recommendations.
_, titles = scann_index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model")

    # Save the index.
    tf.saved_model.save(
        index,
        path,
        options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
    )

    # Load it back; can also be done in TensorFlow Serving.
    loaded = tf.saved_model.load(path)

    # Pass a user id in, get top predicted movie titles back.
    scores, titles = loaded(["42"])

    print(f"Recommendations: {titles[0][:3]}")