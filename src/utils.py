def mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
        
def bounding_box(x_train, x_test, fragment_control=True):
    ad_indxs = []
    if fragment_control:
        mask = x_test.loc[:, [col for col in x_test.columns if col not in x_train.columns]]
        ad_list_fr = mask[mask!=0].dropna(axis=0, how='all').index
    for col in x_train.columns:
        if col in x_test.columns:
            ad_list = list(x_test[(x_test[col] < x_train[col].min()) | (x_test[col] > x_train[col].max())].index)
            if len(ad_list)>0:
                ad_indxs.extend(ad_list)
    if len(ad_indxs) > 0:
        if fragment_control and len(ad_list_fr)>0:
            ad_indxs.extend(ad_list_fr)
        ad_indxs = list(set(ad_indxs))
    return ad_indxs


def fragment_control(x_train, x_test, fragment_control=True):
    ad_indxs = []
    ad_list = []
    if fragment_control:
        mask = x_test.loc[:, [col for col in x_test.columns if col not in x_train.columns]]
        ad_list_fr = mask[mask!=0].dropna(axis=0, how='all').index
    for col in x_train.columns:
        if col in x_test.columns:
            if x_train[col].sum()==0:
                ad_list = list(x_test[(x_test[col]>0)].index)
            if len(ad_list)>0:
                ad_indxs.extend(ad_list)
    if len(ad_indxs) > 0:
        if fragment_control and len(ad_list_fr)>0:
            ad_indxs.extend(ad_list_fr)
        ad_indxs = list(set(ad_indxs))
    return ad_indxs
