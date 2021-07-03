# cell-count-and-segmentation
Either segmentation or cell counting

utils 里面的东西暂时用处不大，写着玩的
之前的dataset object 都被放到consepnaivedataset.py 里面了，因为没有load horizontal map
tianqi写的get_hv_map，只需要开头跑一遍把所有图存下来就好了，大概四分钟
不要频繁的run 那个唯一的.sh file，不然刚刚那四分钟就白跑了
