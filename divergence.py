def compute_js_divergence(p, q, pad_mask=None):
    
    m = 0.5 * (p + q)
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(m, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(m, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum(dim=1).mean()
    q_loss = q_loss.sum(dim=1).mean()

    loss = (p_loss + q_loss) / 2
    return loss