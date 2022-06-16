DEVICE {field} quintic_s_box_{field}({field} l, {field} pre_add, {field} post_add) {{
    {field} tmp = {field}_add(l, pre_add);
    tmp = {field}_sqr(l);
    tmp = {field}_sqr(tmp);
    tmp = {field}_mul(tmp, l);
    tmp = {field}_add(tmp, post_add);

    return tmp;
  }}

DEVICE {field} scalar_product_{field}(CONSTANT {field}* a, {field}* b, int size) {{
    {field} res = {field}_ZERO;

    for (int i = 0; i < size; ++i) {{
        {field} tmp = {field}_mul(a[i], b[i]);
        res = {field}_add(res, tmp);
      }}

    return res;
  }}
