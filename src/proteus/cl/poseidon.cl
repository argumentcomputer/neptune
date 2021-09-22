typedef struct state_{arity}_{strength} {{
  {field} elements[{width}];
  int current_round;
  int rk_offset;
}} state_{arity}_{strength};

DEVICE state_{arity}_{strength} add_round_key_{arity}_{strength} (CONSTANT {field} constants[{constants_elements}], state_{arity}_{strength} s, int i) {{
    s.elements[i] = {field}_add(s.elements[i], (constants + {round_keys_offset})[s.rk_offset + i]);
    return s;
}}

DEVICE state_{arity}_{strength} apply_matrix_{arity}_{strength} (CONSTANT {field} matrix[{width}][{width}], state_{arity}_{strength} s) {{
    {field} tmp[{width}];
    for (int i = 0; i < {width}; ++i) {{
        tmp[i] = s.elements[i];
        s.elements[i] = {field}_ZERO;
      }}

    for (int j = 0; j < {width}; ++j) {{
        for (int i = 0; i < {width}; ++i) {{
            s.elements[j] = {field}_add(s.elements[j], {field}_mul(matrix[i][j], tmp[i]));
          }}
      }}
    return s;
  }}

DEVICE state_{arity}_{strength} apply_sparse_matrix_{arity}_{strength} (CONSTANT {field} sm[{sparse_matrix_size}], state_{arity}_{strength} s) {{
    {field} first_elt = s.elements[0];

    s.elements[0] = scalar_product(sm + {w_hat_offset}, s.elements, {width});

    for (int i = 1; i < {width}; ++i) {{
        {field} val = {field}_mul((sm + {v_rest_offset})[i-1], first_elt);
        s.elements[i] = {field}_add(s.elements[i], val);
      }}

    return s;
  }}

DEVICE state_{arity}_{strength} apply_round_matrix_{arity}_{strength} (CONSTANT {field} constants[{constants_elements}], state_{arity}_{strength} s) {{
    // Needed to get the casting right
    typedef {field} {field}Matrix[{width}][{width}];
    if (s.current_round == {sparse_offset}) {{
#ifdef CUDA
        s = apply_matrix_{arity}_{strength}(*({field}Matrix *)(constants + {pre_sparse_matrix_offset}), s);
#else
        s = apply_matrix_{arity}_{strength}(constants + {pre_sparse_matrix_offset}, s);
#endif
      }} else if ((s.current_round > {sparse_offset}) && (s.current_round < {full_half} + {partial_rounds})) {{
        int index = s.current_round - {sparse_offset} - 1;
        s = apply_sparse_matrix_{arity}_{strength}(constants + {sparse_matrixes_offset} + (index * {sparse_matrix_size}), s);
      }} else {{
#ifdef CUDA
        s = apply_matrix_{arity}_{strength}(*({field}Matrix *)(constants + {mds_matrix_offset}), s);
#else
        s = apply_matrix_{arity}_{strength}(constants + {mds_matrix_offset}, s);
#endif
      }}
    return s;
  }}

DEVICE state_{arity}_{strength} add_full_round_keys_{arity}_{strength} (CONSTANT {field} constants[{constants_elements}], state_{arity}_{strength} s) {{
    for (int i = 0; i < {width}; ++i) {{
        s = add_round_key_{arity}_{strength}(constants, s, i);
      }}
    s.rk_offset += {width};
    return s;
  }}

/* Unused, kept for completeness
DEVICE state_{arity}_{strength} add_partial_round_key_{arity}_{strength} (CONSTANT {field} constants[{constants_elements}], state_{arity}_{strength} s) {{
    s = add_round_key_{arity}_{strength}(constants, s, 0);
    s.rk_offset += 1;
    return s;
}}
*/

DEVICE state_{arity}_{strength} full_round_{arity}_{strength} (CONSTANT {field} constants[{constants_elements}], state_{arity}_{strength} s) {{
    for (int i = 0; i < {width}; ++i) {{
        s.elements[i] = quintic_s_box(s.elements[i], {field}_ZERO, (constants + {round_keys_offset})[s.rk_offset + i]);
      }}
    s.rk_offset += {width};
    s = apply_round_matrix_{arity}_{strength}(constants, s);
    s.current_round += 1;
    return s;
}}

DEVICE state_{arity}_{strength} last_full_round_{arity}_{strength} (CONSTANT {field} constants[{constants_elements}], state_{arity}_{strength} s) {{
    for (int i = 0; i < {width}; ++i) {{
        s.elements[i] = quintic_s_box(s.elements[i], {field}_ZERO, {field}_ZERO);
      }}
    s = apply_round_matrix_{arity}_{strength}(constants, s);
    return s;
}}

DEVICE state_{arity}_{strength} partial_round_{arity}_{strength} (CONSTANT {field} constants[{constants_elements}], state_{arity}_{strength} s) {{
    s.elements[0] = quintic_s_box(s.elements[0], {field}_ZERO, (constants + {round_keys_offset})[s.rk_offset]);
    s.rk_offset += 1;
    s = apply_round_matrix_{arity}_{strength}(constants, s);
    s.current_round += 1;
    return s;
}}

DEVICE state_{arity}_{strength} hash_{arity}_{strength} (CONSTANT {field} constants[{constants_elements}], state_{arity}_{strength} s) {{
    s = add_full_round_keys_{arity}_{strength}(constants, s);

    for (int i = 0; i < {full_half}; ++i) {{
        s = full_round_{arity}_{strength}(constants, s);
      }}
    for (int i = 0; i < {partial_rounds}; ++ i) {{
        s = partial_round_{arity}_{strength}(constants, s);
      }}
    for (int i = 0; i < ({full_half} - 1); ++ i) {{
        s = full_round_{arity}_{strength}(constants, s);
      }}
    s = last_full_round_{arity}_{strength}(constants, s);

    return s;
  }}

KERNEL void hash_preimages_{arity}_{strength}(CONSTANT {field} constants[{constants_elements}],
                             GLOBAL {field} *preimages,
                             GLOBAL {field} *digests,
                             int batch_size
                             ) {{
    int global_id = GET_GLOBAL_ID();

    if (global_id < batch_size) {{
        int offset = global_id * {arity};

        state_{arity}_{strength} s;
        s.elements[0] = constants[{domain_tag_offset}];
        for (int i = 0; i < {arity}; ++i) {{
            s.elements[i+1] = preimages[offset + i];
          }}
        s.current_round = 0;
        s.rk_offset = 0;

        s = hash_{arity}_{strength}(constants, s);

        digests[global_id] = s.elements[1];
      }}
  }}
