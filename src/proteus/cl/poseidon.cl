typedef struct state {{
  {field} elements[{width}];
  int current_round;
  int rk_offset;
}} state;

void debug_f({field} f) {{
    {field}_print({field}_unmont(f));
    printf("\n");
}}
void debug(state s) {{
    if (get_global_id(0) == 0) {{
        printf("state: ");
        for (int i = 0; i < {width}; ++i) {{
            {field} x = s.elements[i];
            debug_f(x);
          }}
        printf("\n");
      }}
}}

void debug_vec(__constant {field} v[], int size) {{
    if (get_global_id(0) == 0) {{
        for (int i = 0; i < size; ++i) {{
            {field} x = v[i];
            debug_f(x);
          }}
        printf("\n");
      }}
}}

{field} quintic_s_box({field} l, {field} pre_add, {field} post_add) {{
    {field} tmp = {field}_add(l, pre_add);
    tmp = {field}_sqr(l);
    tmp = {field}_sqr(tmp);
    tmp = {field}_mul(tmp, l);
    tmp = {field}_add(tmp, post_add); 

    return tmp;
  }}

state add_round_key(__constant {field} constants[{constants_elements}], state s, int i) {{
    s.elements[i] = {field}_add(s.elements[i], (constants + {round_keys_offset})[s.rk_offset + i]);
    return s;
}}

state apply_matrix (__constant {field} matrix[{width}][{width}], state s) {{
    {field} tmp[{width}];
    for (int i = 0; i < {width}; ++i) {{
        tmp[i] = s.elements[i];
        s.elements[i] = {field}_ZERO;
      }}

    int size = {width}*{width};
    for (int j = 0; j < {width}; ++j) {{
        for (int i = 0; i < {width}; ++i) {{
            s.elements[j] = {field}_add(s.elements[j], {field}_mul(matrix[i][j], tmp[i]));
          }}
      }}
    return s;
  }}

{field} scalar_product(__constant {field}* a, {field}* b, int size) {{
    {field} res = {field}_ZERO;

    for (int i = 0; i < size; ++i) {{
        {field} tmp = {field}_mul(a[i], b[i]);
        res = {field}_add(res, tmp);
      }}

    return res;
  }}

state apply_sparse_matrix (__constant {field} sm[{sparse_matrix_size}], state s) {{
    {field} first_elt = s.elements[0];

    s.elements[0] = scalar_product(sm + {w_hat_offset}, s.elements, {width});

    for (int i = 1; i < {width}; ++i) {{
        {field} val = {field}_mul((sm + {v_rest_offset})[i-1], first_elt);
        s.elements[i] = {field}_add(s.elements[i], val);
      }}

    return s;
  }}

state apply_round_matrix (__constant {field} constants[{constants_elements}], state s) {{
    if (s.current_round == {sparse_offset}) {{
        s = apply_matrix(constants + {pre_sparse_matrix_offset}, s);
      }} else if ((s.current_round > {sparse_offset}) && (s.current_round < {full_half} + {partial_rounds})) {{
        int index = s.current_round - {sparse_offset} - 1;
        s = apply_sparse_matrix(constants + {sparse_matrixes_offset} + (index * {sparse_matrix_size}), s);
      }} else {{
        s = apply_matrix(constants + {mds_matrix_offset}, s);
      }}
    return s;
  }}

state add_full_round_keys (__constant {field} constants[{constants_elements}], state s) {{
    for (int i = 0; i < {width}; ++i) {{
        s = add_round_key(constants, s, i);
      }}
    s.rk_offset += {width};
    return s;
  }}

state add_partial_round_key (__constant {field} constants[{constants_elements}], state s) {{
    s = add_round_key(constants, s, 0);
    s.rk_offset += 1;
    return s;
}}

state full_round (__constant {field} constants[{constants_elements}], state s) {{
    for (int i = 0; i < {width}; ++i) {{
        s.elements[i] = quintic_s_box(s.elements[i], {field}_ZERO, (constants + {round_keys_offset})[s.rk_offset + i]);
      }}
    s.rk_offset += {width};
    s = apply_round_matrix(constants, s);
    s.current_round += 1;
    return s;
}}

state last_full_round (__constant {field} constants[{constants_elements}], state s) {{
    for (int i = 0; i < {width}; ++i) {{
        s.elements[i] = quintic_s_box(s.elements[i], {field}_ZERO, {field}_ZERO);
      }}
    s = apply_round_matrix(constants, s);
    return s;
}}

state partial_round (__constant {field} constants[{constants_elements}], state s) {{
    s.elements[0] = quintic_s_box(s.elements[0], {field}_ZERO, (constants + {round_keys_offset})[s.rk_offset]);
    s.rk_offset += 1;
    s = apply_round_matrix(constants, s);
    s.current_round += 1;
    return s;
}}

state hash (__constant {field} constants[{constants_elements}], state s) {{
    s = add_full_round_keys(constants, s);

    for (int i = 0; i < {full_half}; ++i) {{
        s = full_round(constants, s);
      }}
    for (int i = 0; i < {partial_rounds}; ++ i) {{
        s = partial_round(constants, s);
      }}
    for (int i = 0; i < ({full_half} - 1); ++ i) {{
        s = full_round(constants, s);
      }}
    s = last_full_round(constants, s);

    return s;
  }}

__kernel void hash_preimages(__constant {field} constants[{constants_elements}],
                             __global {field} *preimages,
                             __global {field} *digests,
                             int batch_size
                             ) {{
    int global_id = get_global_id(0);

    if (global_id < batch_size) {{
        int offset = global_id * {arity};

        state s;
        s.elements[0] = constants[{domain_tag_offset}];
        for (int i = 0; i < {arity}; ++i) {{
            s.elements[i+1] = preimages[offset + i];
          }}
        s.current_round = 0;
        s.rk_offset = 0;

        s = hash(constants, s);

        digests[global_id] = s.elements[1];
      }}
  }}
