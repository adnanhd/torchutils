#!/usr/bin/env python3

def loss_to_metric(loss):
    def metric(y_true, y_pred):
        return loss(input=y_pred, target=y_true)

    return metric


def one_hot_decode(score):
    def decoded_score(y_pred, y_true):
        return score(y_pred=y_pred.argmax(axis=1), y_true=y_true.argmax(axis=1))

    return decoded_score
