def transfer_img(cfg,img_s,img_t,fg_s,fg_t,bg_s,bg_t):
    '''
    generate two kinds of images,
        - source fg + target bg
        - target fg + source bg
    '''
    rand_st_img = torch.rand_like(img_s)
    for idx,img in enumerate(rang_st_img):
        rang_st_img[idx] = get_a2b_img(img_s[idx],fg_s[idx],bg_s[idx],img_t[idx])
    rand_ts_img = torch.rand_like(img_t)
    for idx,img in enumerate(rang_ts_img):
        rang_ts_img[idx] = get_a2b_img(img_t[idx],fg_t[idx],bg_t[idx],img_s[idx])    
    return rand_st_img,rand_ts_img

def get_a2b_img(img_a,fg_a,bg_a,img_b):
    return img_b*bg_a + img_a*fg_a
