import tensorflow as tf

with tf.variable_scope("root"):
    # At start, the scope is not reusing.
    assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo"):
        # Opened a sub-scope, still not reusing.
        assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo", reuse=True):
        # Explicitly opened a reusing scope.
        assert tf.get_variable_scope().reuse == True
        with tf.variable_scope("bar"):
            # Now sub-scope inherits the reuse flag.
            assert tf.get_variable_scope().reuse == True

    # Exited the reusing scope, back to a non-reusing one.
    assert tf.get_variable_scope().reuse == False

with tf.variable_scope("foo") as foo_scope:
  v = tf.get_variable("v", [1])
with tf.variable_scope(foo_scope):
  w = tf.get_variable("w", [1])
with tf.variable_scope(foo_scope, reuse=True):
  v1 = tf.get_variable("v", [1])
  w1 = tf.get_variable("w", [1])
assert v1 == v
assert w1 == w