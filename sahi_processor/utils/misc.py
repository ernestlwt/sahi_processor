def split_list_into_batches(my_list, batchsize):
    l = len(my_list)
    for i in range(0, l, batchsize):
        yield my_list[i:min(i+batchsize, l)]
