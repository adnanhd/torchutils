import torchutils


x = torchutils.callbacks.EarlyStopping(monitor='foo', goal='minimize')
y = torchutils.callbacks.LogMetersCallback()
y.register_score_name('bar')
x.register_score_name('foo')
cl = torchutils.callbacks.CallbackHandler([x, y])
scores_names = cl.get_score_names()