import acl
r = acl.init(); print('init', r, flush=True)
r = acl.rt.set_device(0); print('dev', r, flush=True)
ctx, r = acl.rt.create_context(0); print('ctx', r, flush=True)
mid, r = acl.mdl.load_from_file('/work/rtdetr_l_orig_310p_linux_aarch64.om'); print('load', r, flush=True)
desc = acl.mdl.create_desc(); r = acl.mdl.get_desc(mid, desc); print('desc', r, flush=True)
