str = "det = (\nmat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[2, 1] * mat[1, 2])\n- mat[0, 1] * (mat[1, 0] * mat[2, 2] - mat[1, 2] * mat[2, 0])\n+ mat[0, 2] * (mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0])\n)"


# A[0][0] = tmpQuad[0]
# A[0][1] = tmpQuad[1]
# A[1][0] = tmpQuad[1]
# A[0][2] = tmpQuad[2]
# A[2][0] = tmpQuad[2]
# A[1][1] = tmpQuad[4]
# A[1][2] = tmpQuad[5]
# A[2][1] = tmpQuad[5]
# A[2][2] = tmpQuad[7]

# b[0] = -1 * tmpQuad[3]
# b[1] = -1 * tmpQuad[6]
# b[2] = -1 * tmpQuad[8]

replacements_0 = {
    "mat[0, 0]": "tmpQuad[0]",
    "mat[0, 1]": "tmpQuad[1]",
    "mat[0, 2]": "tmpQuad[2]",
    "mat[1, 0]": "tmpQuad[1]",
    "mat[1, 1]": "tmpQuad[4]",
    "mat[1, 2]": "tmpQuad[5]",
    "mat[2, 0]": "tmpQuad[2]",
    "mat[2, 1]": "tmpQuad[5]",
    "mat[2, 2]": "tmpQuad[7]",
}

replacements_1 = {
    "mat[0, 0]": "-tmpQuad[3]",
    "mat[0, 1]": "tmpQuad[1]",
    "mat[0, 2]": "tmpQuad[2]",
    "mat[1, 0]": "-tmpQuad[6]",
    "mat[1, 1]": "tmpQuad[4]",
    "mat[1, 2]": "tmpQuad[5]",
    "mat[2, 0]": "-tmpQuad[8]",
    "mat[2, 1]": "tmpQuad[5]",
    "mat[2, 2]": "tmpQuad[7]",
}

replacements_2 = {
    "mat[0, 0]": "tmpQuad[0]",
    "mat[0, 1]": "-tmpQuad[3]",
    "mat[0, 2]": "tmpQuad[2]",
    "mat[1, 0]": "tmpQuad[1]",
    "mat[1, 1]": "-tmpQuad[6]",
    "mat[1, 2]": "tmpQuad[5]",
    "mat[2, 0]": "tmpQuad[2]",
    "mat[2, 1]": "-tmpQuad[8]",
    "mat[2, 2]": "tmpQuad[7]",
}

replacements_3 = {
    "mat[0, 0]": "tmpQuad[0]",
    "mat[0, 1]": "tmpQuad[1]",
    "mat[0, 2]": "-tmpQuad[3]",
    "mat[1, 0]": "tmpQuad[1]",
    "mat[1, 1]": "tmpQuad[4]",
    "mat[1, 2]": "-tmpQuad[6]",
    "mat[2, 0]": "tmpQuad[2]",
    "mat[2, 1]": "tmpQuad[5]",
    "mat[2, 2]": "-tmpQuad[8]",
}


def print_with_replacements(str, replacements):
    for key in replacements:
        str = str.replace(key, replacements[key])
    print(str)


print_with_replacements(str, replacements_0)
print()
print_with_replacements(str, replacements_1)
print()
print_with_replacements(str, replacements_2)
print()
print_with_replacements(str, replacements_3)
print()
