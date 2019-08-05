# from numpy import exp
from math import exp  # math throws better exceptions than numpy


def grad_log_norm_symbolic_2(x1, x2):
    t2 = x1 - x2
    t3 = 1.0 / t2
    t4 = exp(x1)
    t5 = t3 * t4
    t6 = exp(x2)
    t7 = 1.0 / t2 ** 2
    t10 = t3 * t6
    t8 = t5 - t10
    t9 = 1.0 / t8
    t11 = t6 * t7
    expression = [t9 * (t5 + t11 - t4 * t7), -t9 * (t10 + t11 - t4 * t7)]
    return expression


def grad_log_norm_symbolic_3(x1, x2, x3):
    t2 = x1 - x2
    t3 = 1.0 / t2
    t4 = x1 - x3
    t5 = 1.0 / t4
    t6 = x2 - x3
    t7 = 1.0 / t6
    t8 = exp(x1)
    t9 = t3 * t5 * t8
    t10 = exp(x2)
    t11 = 1.0 / t2 ** 2
    t12 = exp(x3)
    t13 = 1.0 / t4 ** 2
    t14 = t5 * t7 * t12
    t18 = t3 * t7 * t10
    t15 = t9 + t14 - t18
    t16 = 1.0 / t15
    t17 = t5 * t8 * t11
    t19 = 1.0 / t6 ** 2
    t20 = t3 * t8 * t13
    t21 = t5 * t12 * t19
    t22 = t7 * t12 * t13
    expression = [-t16 * (-t9 + t17 + t20 + t22 - t7 * t10 * t11), -t16 * (
            -t17 + t18 + t21 + t7 * t10 * t11 - t3 * t10 * t19),
                  t16 * (t14 + t20 + t21 + t22 - t3 * t10 * t19)]
    return expression


def grad_log_norm_symbolic_4(x1, x2, x3, x4):
    t2 = x1 - x2
    t3 = 1.0 / t2
    t4 = x1 - x3
    t5 = 1.0 / t4
    t6 = x2 - x3
    t7 = 1.0 / t6
    t8 = x1 - x4
    t9 = 1.0 / t8
    t10 = x2 - x4
    t11 = 1.0 / t10
    t12 = x3 - x4
    t13 = 1.0 / t12
    t14 = exp(x1)
    t15 = t3 * t5 * t9 * t14
    t16 = exp(x2)
    t17 = 1.0 / t2 ** 2
    t18 = exp(x3)
    t19 = 1.0 / t4 ** 2
    t20 = exp(x4)
    t21 = 1.0 / t8 ** 2
    t22 = t5 * t7 * t13 * t18
    t26 = t3 * t7 * t11 * t16
    t29 = t9 * t11 * t13 * t20
    t23 = t15 + t22 - t26 - t29
    t24 = 1.0 / t23
    t25 = t5 * t9 * t14 * t17
    t27 = 1.0 / t6 ** 2
    t28 = 1.0 / t10 ** 2
    t30 = t3 * t9 * t14 * t19
    t31 = t3 * t11 * t16 * t27
    t32 = t7 * t13 * t18 * t19
    t33 = 1.0 / t12 ** 2
    t34 = t3 * t5 * t14 * t21
    t35 = t3 * t7 * t16 * t28
    t36 = t9 * t11 * t20 * t33
    t37 = t9 * t13 * t20 * t28
    expression = [-t24 * (
            -t15 + t25 + t30 + t32 + t34 - t7 * t11 * t16 * t17 - t11 * t13 * t20 * t21),
                  t24 * (
                          t25 - t26 + t31 + t35 + t37 - t7 * t11 * t16 * t17 - t5 * t13 * t18 * t27),
                  t24 * (
                          t22 + t30 - t31 + t32 + t36 - t5 * t7 * t18 * t33 + t5 * t13 * t18 * t27),
                  -t24 * (
                          t29 - t34 + t35 + t36 + t37 - t5 * t7 * t18 * t33 + t11 * t13 * t20 * t21)]
    return expression


def grad_log_norm_symbolic_5(x1, x2, x3, x4, x5):
    t2 = x1 - x2
    t3 = 1.0 / t2
    t4 = x1 - x3
    t5 = 1.0 / t4
    t6 = x2 - x3
    t7 = 1.0 / t6
    t8 = x1 - x4
    t9 = 1.0 / t8
    t10 = x2 - x4
    t11 = 1.0 / t10
    t12 = x3 - x4
    t13 = 1.0 / t12
    t14 = x1 - x5
    t15 = 1.0 / t14
    t16 = x2 - x5
    t17 = 1.0 / t16
    t18 = x3 - x5
    t19 = 1.0 / t18
    t20 = x4 - x5
    t21 = 1.0 / t20
    t22 = exp(x1)
    t23 = t3 * t5 * t9 * t15 * t22
    t24 = exp(x2)
    t25 = 1.0 / t2 ** 2
    t26 = exp(x3)
    t27 = 1.0 / t4 ** 2
    t28 = exp(x4)
    t29 = 1.0 / t8 ** 2
    t30 = exp(x5)
    t31 = 1.0 / t14 ** 2
    t32 = t5 * t7 * t13 * t19 * t26
    t33 = t15 * t17 * t19 * t21 * t30
    t37 = t3 * t7 * t11 * t17 * t24
    t41 = t9 * t11 * t13 * t21 * t28
    t34 = t23 + t32 + t33 - t37 - t41
    t35 = 1.0 / t34
    t36 = t5 * t9 * t15 * t22 * t25
    t38 = 1.0 / t6 ** 2
    t39 = 1.0 / t10 ** 2
    t40 = 1.0 / t16 ** 2
    t42 = t3 * t9 * t15 * t22 * t27
    t43 = t3 * t11 * t17 * t24 * t38
    t44 = t7 * t13 * t19 * t26 * t27
    t45 = 1.0 / t12 ** 2
    t46 = 1.0 / t18 ** 2
    t47 = t3 * t5 * t15 * t22 * t29
    t48 = t3 * t7 * t17 * t24 * t39
    t49 = t9 * t11 * t21 * t28 * t45
    t50 = t9 * t13 * t21 * t28 * t39
    t51 = 1.0 / t20 ** 2
    t52 = t3 * t5 * t9 * t22 * t31
    t53 = t3 * t7 * t11 * t24 * t40
    t54 = t15 * t17 * t19 * t30 * t51
    t55 = t17 * t19 * t21 * t30 * t31
    expression = [-t35 * (
            -t23 + t36 + t42 + t44 + t47 + t52 + t55 - t7 * t11 * t17 * t24 * t25 - t11 * t13 * t21 * t28 * t29),
                  t35 * (
                          t36 - t37 + t43 + t48 + t50 + t53 - t7 * t11 * t17 * t24 * t25 - t5 * t13 * t19 * t26 * t38 - t15 * t19 * t21 * t30 * t40),
                  t35 * (
                          t32 + t42 - t43 + t44 + t49 - t5 * t7 * t13 * t26 * t46 + t5 * t13 * t19 * t26 * t38 - t5 * t7 * t19 * t26 * t45 - t15 * t17 * t21 * t30 * t46),
                  -t35 * (
                          t41 - t47 + t48 + t49 + t50 + t54 - t5 * t7 * t19 * t26 * t45 + t11 * t13 * t21 * t28 * t29 - t9 * t11 * t13 * t28 * t51),
                  t35 * (
                          t33 + t52 - t53 + t54 + t55 + t5 * t7 * t13 * t26 * t46 - t9 * t11 * t13 * t28 * t51 + t15 * t19 * t21 * t30 * t40 + t15 * t17 * t21 * t30 * t46)]
    return expression



def grad_log_norm_symbolic_6(x1, x2, x3, x4, x5, x6):
    t2 = x1 - x2
    t3 = 1.0 / t2
    t4 = x1 - x3
    t5 = 1.0 / t4
    t6 = x2 - x3
    t7 = 1.0 / t6
    t8 = x1 - x4
    t9 = 1.0 / t8
    t10 = x2 - x4
    t11 = 1.0 / t10
    t12 = x3 - x4
    t13 = 1.0 / t12
    t14 = x1 - x5
    t15 = 1.0 / t14
    t16 = x2 - x5
    t17 = 1.0 / t16
    t18 = x3 - x5
    t19 = 1.0 / t18
    t20 = x4 - x5
    t21 = 1.0 / t20
    t22 = x1 - x6
    t23 = 1.0 / t22
    t24 = x2 - x6
    t25 = 1.0 / t24
    t26 = x3 - x6
    t27 = 1.0 / t26
    t28 = x4 - x6
    t29 = 1.0 / t28
    t30 = x5 - x6
    t31 = 1.0 / t30
    t32 = exp(x1)
    t33 = t3 * t5 * t9 * t15 * t23 * t32
    t34 = exp(x2)
    t35 = 1.0 / t2 ** 2
    t36 = exp(x3)
    t37 = 1.0 / t4 ** 2
    t38 = exp(x4)
    t39 = 1.0 / t8 ** 2
    t40 = exp(x5)
    t41 = 1.0 / t14 ** 2
    t42 = exp(x6)
    t43 = 1.0 / t22 ** 2
    t44 = t5 * t7 * t13 * t19 * t27 * t36
    t45 = t15 * t17 * t19 * t21 * t31 * t40
    t49 = t3 * t7 * t11 * t17 * t25 * t34
    t54 = t9 * t11 * t13 * t21 * t29 * t38
    t55 = t23 * t25 * t27 * t29 * t31 * t42
    t46 = t33 + t44 + t45 - t49 - t54 - t55
    t47 = 1.0 / t46
    t48 = t5 * t9 * t15 * t23 * t32 * t35
    t50 = 1.0 / t6 ** 2
    t51 = 1.0 / t10 ** 2
    t52 = 1.0 / t16 ** 2
    t53 = 1.0 / t24 ** 2
    t56 = t3 * t9 * t15 * t23 * t32 * t37
    t57 = t3 * t11 * t17 * t25 * t34 * t50
    t58 = t7 * t13 * t19 * t27 * t36 * t37
    t59 = 1.0 / t12 ** 2
    t60 = 1.0 / t18 ** 2
    t61 = 1.0 / t26 ** 2
    t62 = t3 * t5 * t15 * t23 * t32 * t39
    t63 = t3 * t7 * t17 * t25 * t34 * t51
    t64 = t9 * t11 * t21 * t29 * t38 * t59
    t65 = t9 * t13 * t21 * t29 * t38 * t51
    t66 = 1.0 / t20 ** 2
    t67 = 1.0 / t28 ** 2
    t68 = t3 * t5 * t9 * t23 * t32 * t41
    t69 = t3 * t7 * t11 * t25 * t34 * t52
    t70 = t15 * t17 * t19 * t31 * t40 * t66
    t71 = t17 * t19 * t21 * t31 * t40 * t41
    t72 = 1.0 / t30 ** 2
    t73 = t3 * t5 * t9 * t15 * t32 * t43
    t74 = t3 * t7 * t11 * t17 * t34 * t53
    t75 = t23 * t25 * t27 * t29 * t42 * t72
    t76 = t23 * t25 * t29 * t31 * t42 * t61
    t77 = t23 * t27 * t29 * t31 * t42 * t53
    
    import numpy as np
    tmp = np.array([
            t2,
            t3,
            t4,
            t5,
            t6,
            t7,
            t8,
            t9,
            t10,
            t11,
            t12,
            t13,
            t14,
            t15,
            t16,
            t17,
            t18,
            t19,
            t20,
            t21,
            t22,
            t23,
            t24,
            t25,
            t26,
            t27,
            t28,
            t29,
            t30,
            t31,
            t32,
            t33,
            t34,
            t35,
            t36,
            t37,
            t38,
            t39,
            t40,
            t41,
            t42,
            t43,
            t44,
            t45,
            t46,
            t47,
            t48,
            t49,
            t50,
            t51,
            t52,
            t53,
            t54,
            t55,
            t56,
            t57,
            t58,
            t59,
            t60,
            t61,
            t62,
            t63,
            t64,
            t65,
            t66,
            t67,
            t68,
            t69,
            t70,
            t71,
            t72,
            t73,
            t74,
            t75,
            t76,
            t77,
        ])

    isfinite = np.isfinite(tmp)
    assert isfinite.all(), ((x1, x2, x3, x4, x5, x6), (t38), (np.nonzero(np.logical_not(isfinite))), tmp[np.nonzero(np.logical_not(isfinite))])
    
    expression = [-t47 * (
            -t33 + t48 + t56 + t58 + t62 + t68 + t71 + t73 - t7 * t11 * t17 * t25 * t34 * t35 - t11 * t13 * t21 * t29 * t38 * t39 - t25 * t27 * t29 * t31 * t42 * t43),
                  t47 * (
                          t48 - t49 + t57 + t63 + t65 + t69 + t74 + t77 - t7 * t11 * t17 * t25 * t34 * t35 - t5 * t13 * t19 * t27 * t36 * t50 - t15 * t19 * t21 * t31 * t40 * t52),
                  t47 * (
                          t44 + t56 - t57 + t58 + t64 + t76 - t5 * t7 * t13 * t19 * t36 * t61 - t5 * t7 * t13 * t27 * t36 * t60 + t5 * t13 * t19 * t27 * t36 * t50 - t5 * t7 * t19 * t27 * t36 * t59 - t15 * t17 * t21 * t31 * t40 * t60),
                  -t47 * (
                          t54 - t62 + t63 + t64 + t65 + t70 + t11 * t13 * t21 * t29 * t38 * t39 - t5 * t7 * t19 * t27 * t36 * t59 - t9 * t11 * t13 * t21 * t38 * t67 - t9 * t11 * t13 * t29 * t38 * t66 - t23 * t25 * t27 * t31 * t42 * t67),
                  t47 * (
                          t45 + t68 - t69 + t70 + t71 + t75 + t5 * t7 * t13 * t27 * t36 * t60 - t9 * t11 * t13 * t29 * t38 * t66 + t15 * t19 * t21 * t31 * t40 * t52 - t15 * t17 * t19 * t21 * t40 * t72 + t15 * t17 * t21 * t31 * t40 * t60),
                  -t47 * (
                          t55 - t73 + t74 + t75 + t76 + t77 - t5 * t7 * t13 * t19 * t36 * t61 + t9 * t11 * t13 * t21 * t38 * t67 - t15 * t17 * t19 * t21 * t40 * t72 + t25 * t27 * t29 * t31 * t42 * t43 + t23 * t25 * t27 * t31 * t42 * t67)]

    return expression


grad_log_norm_symbolic = {
    2: grad_log_norm_symbolic_2,
    3: grad_log_norm_symbolic_3,
    4: grad_log_norm_symbolic_4,
    5: grad_log_norm_symbolic_5,
    6: grad_log_norm_symbolic_6,
}


def grad_log_norm_symbolic_diff_2(x1, x2):
    return grad_log_norm_symbolic_2(
        x1 + x2,
        x2,
    )


def grad_log_norm_symbolic_diff_3(x1, x2, x3):
    return grad_log_norm_symbolic_3(
        x1 + x2 + x3,
        x2 + x3,
        x3
    )


def grad_log_norm_symbolic_diff_4(x1, x2, x3, x4):
    return grad_log_norm_symbolic_4(
        x1 + x2 + x3 + x4,
        x2 + x3 + x4,
        x3 + x4,
        x4,
    )
def grad_log_norm_symbolic_diff_5(x1, x2, x3, x4, x5):
    return grad_log_norm_symbolic_5(
        x1 + x2 + x3 + x4 + x5,
        x2 + x3 + x4 + x5,
        x3 + x4 + x5,
        x4 + x5,
        x5,
    )

def grad_log_norm_symbolic_diff_6(x1, x2, x3, x4, x5, x6):
    return grad_log_norm_symbolic_6(
        x1 + x2 + x3 + x4 + x5 + x6,
        x2 + x3 + x4 + x5 + x6,
        x3 + x4 + x5 + x6,
        x4 + x5 + x6,
        x5 + x6,
        x6,
    )


grad_log_norm_symbolic_diff = {
    2: grad_log_norm_symbolic_diff_2,
    3: grad_log_norm_symbolic_diff_3,
    4: grad_log_norm_symbolic_diff_4,
    5: grad_log_norm_symbolic_diff_5,
    6: grad_log_norm_symbolic_diff_6,
}