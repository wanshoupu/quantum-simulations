import textwrap
from textwrap import dedent

import sympy
from sympy import Matrix, symbols, kronecker_product as kron, pprint
from sympy.printing.pretty import pretty

from sandbox.sym.inter_product import inter_product, mesh_product
from sandbox.sym.sym_gen import symmat


def test_kron():
    # Define symbolic variables
    a, b, c, d = symbols('a b c d')
    e, f, g, h = symbols('e f g h')

    # Define two symbolic matrices
    A = Matrix([[a, b], [c, d]])
    B = sympy.eye(2)
    C = Matrix([[e, f], [g, h]])

    # Compute the Kronecker product
    K = kron(A, B, C)

    # print()
    # pprint(K)
    expected = """
        ⎡a⋅e  a⋅f   0    0   b⋅e  b⋅f   0    0 ⎤
        ⎢                                      ⎥
        ⎢a⋅g  a⋅h   0    0   b⋅g  b⋅h   0    0 ⎥
        ⎢                                      ⎥
        ⎢ 0    0   a⋅e  a⋅f   0    0   b⋅e  b⋅f⎥
        ⎢                                      ⎥
        ⎢ 0    0   a⋅g  a⋅h   0    0   b⋅g  b⋅h⎥
        ⎢                                      ⎥
        ⎢c⋅e  c⋅f   0    0   d⋅e  d⋅f   0    0 ⎥
        ⎢                                      ⎥
        ⎢c⋅g  c⋅h   0    0   d⋅g  d⋅h   0    0 ⎥
        ⎢                                      ⎥
        ⎢ 0    0   c⋅e  c⋅f   0    0   d⋅e  d⋅f⎥
        ⎢                                      ⎥
        ⎣ 0    0   c⋅g  c⋅h   0    0   d⋅g  d⋅h⎦
    """
    assert pretty(K) == textwrap.dedent(expected).strip()


def test_kron_4qubit_sandwich():
    """
    test the configuration of unitary matrix acting on first and third qubit with the second qubit unchanged.
    """
    # Define symbolic variables
    A = symmat(2)
    # print('A')
    # pprint(A)
    B = sympy.eye(8)
    swap = Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    K = kron(A, B)
    # print('K')
    # pprint(K)

    s12 = kron(sympy.eye(2), sympy.eye(2), swap, )
    final = s12 * K * s12

    # print('final')
    # pprint(final)
    assert K == final


def test_inter_product():
    """
    test the configuration of Kronecker product A ⨁ I ⨁ C
    """
    coms = symmat(5, 'a'), symmat(3, 'c')
    A = kron(*coms)
    B = sympy.eye(2)

    expected = kron(coms[0], B, coms[1])
    # pprint(expected, num_columns=10000)
    assert inter_product(A, B, 3) == expected


def test_sandwich_product_arbitray_matrix():
    """
        let m = len(A), n = len(I), l = len(C)
        then the Kproduct, A ⨁ I ⨁ C, is formed by
        1. get the Kproduct K = A ⨁ C
        2. divide up K
    """
    # Define symbolic variables
    A = symmat(15, 'a')
    # print('A')
    # pprint(A, num_columns=10000)

    B = symmat(2, 'b')
    # print('B')
    # pprint(B)

    C = inter_product(A, B, 5)
    # print('C')
    # pprint(C, num_columns=10000)

    expected = """
        ⎡a⋅aa  a⋅ab  a⋅ac  a⋅ad  a⋅ae  aa⋅b  ab⋅b  ac⋅b  ad⋅b  ae⋅b  a⋅af  a⋅ag  a⋅ah  a⋅ai  a⋅aj  af⋅b  ag⋅b  ah⋅b  ai⋅b  aj⋅b  a⋅ak  a⋅al  a⋅am  a⋅an  a⋅ao  ak⋅b  al⋅b  am⋅b  an⋅b  ao⋅b⎤
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅ap  a⋅aq  a⋅ar  a⋅as  a⋅at  ap⋅b  aq⋅b  ar⋅b  as⋅b  at⋅b  a⋅au  a⋅av  a⋅aw  a⋅ax  a⋅ay  au⋅b  av⋅b  aw⋅b  ax⋅b  ay⋅b  a⋅az  a⋅ba  a⋅bb  a⋅bc  a⋅bd  az⋅b  b⋅ba  b⋅bb  b⋅bc  b⋅bd⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅be  a⋅bf  a⋅bg  a⋅bh  a⋅bi  b⋅be  b⋅bf  b⋅bg  b⋅bh  b⋅bi  a⋅bj  a⋅bk  a⋅bl  a⋅bm  a⋅bn  b⋅bj  b⋅bk  b⋅bl  b⋅bm  b⋅bn  a⋅bo  a⋅bp  a⋅bq  a⋅br  a⋅bs  b⋅bo  b⋅bp  b⋅bq  b⋅br  b⋅bs⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅bt  a⋅bu  a⋅bv  a⋅bw  a⋅bx  b⋅bt  b⋅bu  b⋅bv  b⋅bw  b⋅bx  a⋅by  a⋅bz  a⋅ca  a⋅cb  a⋅cc  b⋅by  b⋅bz  b⋅ca  b⋅cb  b⋅cc  a⋅cd  a⋅ce  a⋅cf  a⋅cg  a⋅ch  b⋅cd  b⋅ce  b⋅cf  b⋅cg  b⋅ch⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅ci  a⋅cj  a⋅ck  a⋅cl  a⋅cm  b⋅ci  b⋅cj  b⋅ck  b⋅cl  b⋅cm  a⋅cn  a⋅co  a⋅cp  a⋅cq  a⋅cr  b⋅cn  b⋅co  b⋅cp  b⋅cq  b⋅cr  a⋅cs  a⋅ct  a⋅cu  a⋅cv  a⋅cw  b⋅cs  b⋅ct  b⋅cu  b⋅cv  b⋅cw⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢aa⋅c  ab⋅c  ac⋅c  ad⋅c  ae⋅c  aa⋅d  ab⋅d  ac⋅d  ad⋅d  ae⋅d  af⋅c  ag⋅c  ah⋅c  ai⋅c  aj⋅c  af⋅d  ag⋅d  ah⋅d  ai⋅d  aj⋅d  ak⋅c  al⋅c  am⋅c  an⋅c  ao⋅c  ak⋅d  al⋅d  am⋅d  an⋅d  ao⋅d⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢ap⋅c  aq⋅c  ar⋅c  as⋅c  at⋅c  ap⋅d  aq⋅d  ar⋅d  as⋅d  at⋅d  au⋅c  av⋅c  aw⋅c  ax⋅c  ay⋅c  au⋅d  av⋅d  aw⋅d  ax⋅d  ay⋅d  az⋅c  ba⋅c  bb⋅c  bc⋅c  bd⋅c  az⋅d  ba⋅d  bb⋅d  bc⋅d  bd⋅d⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢be⋅c  bf⋅c  bg⋅c  bh⋅c  bi⋅c  be⋅d  bf⋅d  bg⋅d  bh⋅d  bi⋅d  bj⋅c  bk⋅c  bl⋅c  bm⋅c  bn⋅c  bj⋅d  bk⋅d  bl⋅d  bm⋅d  bn⋅d  bo⋅c  bp⋅c  bq⋅c  br⋅c  bs⋅c  bo⋅d  bp⋅d  bq⋅d  br⋅d  bs⋅d⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢bt⋅c  bu⋅c  bv⋅c  bw⋅c  bx⋅c  bt⋅d  bu⋅d  bv⋅d  bw⋅d  bx⋅d  by⋅c  bz⋅c  c⋅ca  c⋅cb  c⋅cc  by⋅d  bz⋅d  ca⋅d  cb⋅d  cc⋅d  c⋅cd  c⋅ce  c⋅cf  c⋅cg  c⋅ch  cd⋅d  ce⋅d  cf⋅d  cg⋅d  ch⋅d⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢c⋅ci  c⋅cj  c⋅ck  c⋅cl  c⋅cm  ci⋅d  cj⋅d  ck⋅d  cl⋅d  cm⋅d  c⋅cn  c⋅co  c⋅cp  c⋅cq  c⋅cr  cn⋅d  co⋅d  cp⋅d  cq⋅d  cr⋅d  c⋅cs  c⋅ct  c⋅cu  c⋅cv  c⋅cw  cs⋅d  ct⋅d  cu⋅d  cv⋅d  cw⋅d⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅cx  a⋅cy  a⋅cz  a⋅da  a⋅db  b⋅cx  b⋅cy  b⋅cz  b⋅da  b⋅db  a⋅dc  a⋅dd  a⋅de  a⋅df  a⋅dg  b⋅dc  b⋅dd  b⋅de  b⋅df  b⋅dg  a⋅dh  a⋅di  a⋅dj  a⋅dk  a⋅dl  b⋅dh  b⋅di  b⋅dj  b⋅dk  b⋅dl⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅dm  a⋅dn  a⋅do  a⋅dp  a⋅dq  b⋅dm  b⋅dn  b⋅do  b⋅dp  b⋅dq  a⋅dr  a⋅ds  a⋅dt  a⋅du  a⋅dv  b⋅dr  b⋅ds  b⋅dt  b⋅du  b⋅dv  a⋅dw  a⋅dx  a⋅dy  a⋅dz  a⋅ea  b⋅dw  b⋅dx  b⋅dy  b⋅dz  b⋅ea⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅eb  a⋅ec  a⋅ed  a⋅ee  a⋅ef  b⋅eb  b⋅ec  b⋅ed  b⋅ee  b⋅ef  a⋅eg  a⋅eh  a⋅ei  a⋅ej  a⋅ek  b⋅eg  b⋅eh  b⋅ei  b⋅ej  b⋅ek  a⋅el  a⋅em  a⋅en  a⋅eo  a⋅ep  b⋅el  b⋅em  b⋅en  b⋅eo  b⋅ep⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅eq  a⋅er  a⋅es  a⋅et  a⋅eu  b⋅eq  b⋅er  b⋅es  b⋅et  b⋅eu  a⋅ev  a⋅ew  a⋅ex  a⋅ey  a⋅ez  b⋅ev  b⋅ew  b⋅ex  b⋅ey  b⋅ez  a⋅fa  a⋅fb  a⋅fc  a⋅fd  a⋅fe  b⋅fa  b⋅fb  b⋅fc  b⋅fd  b⋅fe⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅ff  a⋅fg  a⋅fh  a⋅fi  a⋅fj  b⋅ff  b⋅fg  b⋅fh  b⋅fi  b⋅fj  a⋅fk  a⋅fl  a⋅fm  a⋅fn  a⋅fo  b⋅fk  b⋅fl  b⋅fm  b⋅fn  b⋅fo  a⋅fp  a⋅fq  a⋅fr  a⋅fs  a⋅ft  b⋅fp  b⋅fq  b⋅fr  b⋅fs  b⋅ft⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢c⋅cx  c⋅cy  c⋅cz  c⋅da  c⋅db  cx⋅d  cy⋅d  cz⋅d  d⋅da  d⋅db  c⋅dc  c⋅dd  c⋅de  c⋅df  c⋅dg  d⋅dc  d⋅dd  d⋅de  d⋅df  d⋅dg  c⋅dh  c⋅di  c⋅dj  c⋅dk  c⋅dl  d⋅dh  d⋅di  d⋅dj  d⋅dk  d⋅dl⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢c⋅dm  c⋅dn  c⋅do  c⋅dp  c⋅dq  d⋅dm  d⋅dn  d⋅do  d⋅dp  d⋅dq  c⋅dr  c⋅ds  c⋅dt  c⋅du  c⋅dv  d⋅dr  d⋅ds  d⋅dt  d⋅du  d⋅dv  c⋅dw  c⋅dx  c⋅dy  c⋅dz  c⋅ea  d⋅dw  d⋅dx  d⋅dy  d⋅dz  d⋅ea⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢c⋅eb  c⋅ec  c⋅ed  c⋅ee  c⋅ef  d⋅eb  d⋅ec  d⋅ed  d⋅ee  d⋅ef  c⋅eg  c⋅eh  c⋅ei  c⋅ej  c⋅ek  d⋅eg  d⋅eh  d⋅ei  d⋅ej  d⋅ek  c⋅el  c⋅em  c⋅en  c⋅eo  c⋅ep  d⋅el  d⋅em  d⋅en  d⋅eo  d⋅ep⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢c⋅eq  c⋅er  c⋅es  c⋅et  c⋅eu  d⋅eq  d⋅er  d⋅es  d⋅et  d⋅eu  c⋅ev  c⋅ew  c⋅ex  c⋅ey  c⋅ez  d⋅ev  d⋅ew  d⋅ex  d⋅ey  d⋅ez  c⋅fa  c⋅fb  c⋅fc  c⋅fd  c⋅fe  d⋅fa  d⋅fb  d⋅fc  d⋅fd  d⋅fe⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢c⋅ff  c⋅fg  c⋅fh  c⋅fi  c⋅fj  d⋅ff  d⋅fg  d⋅fh  d⋅fi  d⋅fj  c⋅fk  c⋅fl  c⋅fm  c⋅fn  c⋅fo  d⋅fk  d⋅fl  d⋅fm  d⋅fn  d⋅fo  c⋅fp  c⋅fq  c⋅fr  c⋅fs  c⋅ft  d⋅fp  d⋅fq  d⋅fr  d⋅fs  d⋅ft⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅fu  a⋅fv  a⋅fw  a⋅fx  a⋅fy  b⋅fu  b⋅fv  b⋅fw  b⋅fx  b⋅fy  a⋅fz  a⋅ga  a⋅gb  a⋅gc  a⋅gd  b⋅fz  b⋅ga  b⋅gb  b⋅gc  b⋅gd  a⋅ge  a⋅gf  a⋅gg  a⋅gh  a⋅gi  b⋅ge  b⋅gf  b⋅gg  b⋅gh  b⋅gi⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅gj  a⋅gk  a⋅gl  a⋅gm  a⋅gn  b⋅gj  b⋅gk  b⋅gl  b⋅gm  b⋅gn  a⋅go  a⋅gp  a⋅gq  a⋅gr  a⋅gs  b⋅go  b⋅gp  b⋅gq  b⋅gr  b⋅gs  a⋅gt  a⋅gu  a⋅gv  a⋅gw  a⋅gx  b⋅gt  b⋅gu  b⋅gv  b⋅gw  b⋅gx⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅gy  a⋅gz  a⋅ha  a⋅hb  a⋅hc  b⋅gy  b⋅gz  b⋅ha  b⋅hb  b⋅hc  a⋅hd  a⋅he  a⋅hf  a⋅hg  a⋅hh  b⋅hd  b⋅he  b⋅hf  b⋅hg  b⋅hh  a⋅hi  a⋅hj  a⋅hk  a⋅hl  a⋅hm  b⋅hi  b⋅hj  b⋅hk  b⋅hl  b⋅hm⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅hn  a⋅ho  a⋅hp  a⋅hq  a⋅hr  b⋅hn  b⋅ho  b⋅hp  b⋅hq  b⋅hr  a⋅hs  a⋅ht  a⋅hu  a⋅hv  a⋅hw  b⋅hs  b⋅ht  b⋅hu  b⋅hv  b⋅hw  a⋅hx  a⋅hy  a⋅hz  a⋅ia  a⋅ib  b⋅hx  b⋅hy  b⋅hz  b⋅ia  b⋅ib⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢a⋅ic  a⋅id  a⋅ie  a⋅if  a⋅ig  b⋅ic  b⋅id  b⋅ie  b⋅if  b⋅ig  a⋅ih  a⋅ii  a⋅ij  a⋅ik  a⋅il  b⋅ih  b⋅ii  b⋅ij  b⋅ik  b⋅il  a⋅im  a⋅in  a⋅io  a⋅ip  a⋅iq  b⋅im  b⋅in  b⋅io  b⋅ip  b⋅iq⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢c⋅fu  c⋅fv  c⋅fw  c⋅fx  c⋅fy  d⋅fu  d⋅fv  d⋅fw  d⋅fx  d⋅fy  c⋅fz  c⋅ga  c⋅gb  c⋅gc  c⋅gd  d⋅fz  d⋅ga  d⋅gb  d⋅gc  d⋅gd  c⋅ge  c⋅gf  c⋅gg  c⋅gh  c⋅gi  d⋅ge  d⋅gf  d⋅gg  d⋅gh  d⋅gi⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢c⋅gj  c⋅gk  c⋅gl  c⋅gm  c⋅gn  d⋅gj  d⋅gk  d⋅gl  d⋅gm  d⋅gn  c⋅go  c⋅gp  c⋅gq  c⋅gr  c⋅gs  d⋅go  d⋅gp  d⋅gq  d⋅gr  d⋅gs  c⋅gt  c⋅gu  c⋅gv  c⋅gw  c⋅gx  d⋅gt  d⋅gu  d⋅gv  d⋅gw  d⋅gx⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢c⋅gy  c⋅gz  c⋅ha  c⋅hb  c⋅hc  d⋅gy  d⋅gz  d⋅ha  d⋅hb  d⋅hc  c⋅hd  c⋅he  c⋅hf  c⋅hg  c⋅hh  d⋅hd  d⋅he  d⋅hf  d⋅hg  d⋅hh  c⋅hi  c⋅hj  c⋅hk  c⋅hl  c⋅hm  d⋅hi  d⋅hj  d⋅hk  d⋅hl  d⋅hm⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎢c⋅hn  c⋅ho  c⋅hp  c⋅hq  c⋅hr  d⋅hn  d⋅ho  d⋅hp  d⋅hq  d⋅hr  c⋅hs  c⋅ht  c⋅hu  c⋅hv  c⋅hw  d⋅hs  d⋅ht  d⋅hu  d⋅hv  d⋅hw  c⋅hx  c⋅hy  c⋅hz  c⋅ia  c⋅ib  d⋅hx  d⋅hy  d⋅hz  d⋅ia  d⋅ib⎥
        ⎢                                                                                                                                                                                  ⎥
        ⎣c⋅ic  c⋅id  c⋅ie  c⋅if  c⋅ig  d⋅ic  d⋅id  d⋅ie  d⋅if  d⋅ig  c⋅ih  c⋅ii  c⋅ij  c⋅ik  c⋅il  d⋅ih  d⋅ii  d⋅ij  d⋅ik  d⋅il  c⋅im  c⋅in  c⋅io  c⋅ip  c⋅iq  d⋅im  d⋅in  d⋅io  d⋅ip  d⋅iq⎦
    """
    assert pretty(C, num_columns=10000) == dedent(expected).strip()


def test_inter_product_8_2_4():
    """
        let m = len(A), n = len(I), l = len(C)
        then the Kproduct, A ⨁ I ⨁ C, is formed by
        1. get the Kproduct K = A ⨁ C
        2. divide up K
    """
    # Define symbolic variables
    A = symmat(8, 'a')
    # print('A')
    # pprint(A, num_columns=10000)

    B = symmat(3, 'b')
    # print('B')
    # pprint(B)

    C = inter_product(A, B, 2)
    # print('C')
    # pprint(C, num_columns=10000)
    expected = '''
        ⎡a⋅aa  a⋅ab  aa⋅b  ab⋅b  aa⋅c  ab⋅c  a⋅ac  a⋅ad  ac⋅b  ad⋅b  ac⋅c  ad⋅c  a⋅ae  a⋅af  ae⋅b  af⋅b  ae⋅c  af⋅c  a⋅ag  a⋅ah  ag⋅b  ah⋅b  ag⋅c  ah⋅c⎤
        ⎢                                                                                                                                              ⎥
        ⎢a⋅ai  a⋅aj  ai⋅b  aj⋅b  ai⋅c  aj⋅c  a⋅ak  a⋅al  ak⋅b  al⋅b  ak⋅c  al⋅c  a⋅am  a⋅an  am⋅b  an⋅b  am⋅c  an⋅c  a⋅ao  a⋅ap  ao⋅b  ap⋅b  ao⋅c  ap⋅c⎥
        ⎢                                                                                                                                              ⎥
        ⎢aa⋅d  ab⋅d  aa⋅e  ab⋅e  aa⋅f  ab⋅f  ac⋅d  ad⋅d  ac⋅e  ad⋅e  ac⋅f  ad⋅f  ae⋅d  af⋅d  ae⋅e  af⋅e  ae⋅f  af⋅f  ag⋅d  ah⋅d  ag⋅e  ah⋅e  ag⋅f  ah⋅f⎥
        ⎢                                                                                                                                              ⎥
        ⎢ai⋅d  aj⋅d  ai⋅e  aj⋅e  ai⋅f  aj⋅f  ak⋅d  al⋅d  ak⋅e  al⋅e  ak⋅f  al⋅f  am⋅d  an⋅d  am⋅e  an⋅e  am⋅f  an⋅f  ao⋅d  ap⋅d  ao⋅e  ap⋅e  ao⋅f  ap⋅f⎥
        ⎢                                                                                                                                              ⎥
        ⎢aa⋅g  ab⋅g  aa⋅h  ab⋅h  aa⋅i  ab⋅i  ac⋅g  ad⋅g  ac⋅h  ad⋅h  ac⋅i  ad⋅i  ae⋅g  af⋅g  ae⋅h  af⋅h  ae⋅i  af⋅i  ag⋅g  ah⋅g  ag⋅h  ah⋅h  ag⋅i  ah⋅i⎥
        ⎢                                                                                                                                              ⎥
        ⎢ai⋅g  aj⋅g  ai⋅h  aj⋅h  ai⋅i  aj⋅i  ak⋅g  al⋅g  ak⋅h  al⋅h  ak⋅i  al⋅i  am⋅g  an⋅g  am⋅h  an⋅h  am⋅i  an⋅i  ao⋅g  ap⋅g  ao⋅h  ap⋅h  ao⋅i  ap⋅i⎥
        ⎢                                                                                                                                              ⎥
        ⎢a⋅aq  a⋅ar  aq⋅b  ar⋅b  aq⋅c  ar⋅c  a⋅as  a⋅at  as⋅b  at⋅b  as⋅c  at⋅c  a⋅au  a⋅av  au⋅b  av⋅b  au⋅c  av⋅c  a⋅aw  a⋅ax  aw⋅b  ax⋅b  aw⋅c  ax⋅c⎥
        ⎢                                                                                                                                              ⎥
        ⎢a⋅ay  a⋅az  ay⋅b  az⋅b  ay⋅c  az⋅c  a⋅ba  a⋅bb  b⋅ba  b⋅bb  ba⋅c  bb⋅c  a⋅bc  a⋅bd  b⋅bc  b⋅bd  bc⋅c  bd⋅c  a⋅be  a⋅bf  b⋅be  b⋅bf  be⋅c  bf⋅c⎥
        ⎢                                                                                                                                              ⎥
        ⎢aq⋅d  ar⋅d  aq⋅e  ar⋅e  aq⋅f  ar⋅f  as⋅d  at⋅d  as⋅e  at⋅e  as⋅f  at⋅f  au⋅d  av⋅d  au⋅e  av⋅e  au⋅f  av⋅f  aw⋅d  ax⋅d  aw⋅e  ax⋅e  aw⋅f  ax⋅f⎥
        ⎢                                                                                                                                              ⎥
        ⎢ay⋅d  az⋅d  ay⋅e  az⋅e  ay⋅f  az⋅f  ba⋅d  bb⋅d  ba⋅e  bb⋅e  ba⋅f  bb⋅f  bc⋅d  bd⋅d  bc⋅e  bd⋅e  bc⋅f  bd⋅f  be⋅d  bf⋅d  be⋅e  bf⋅e  be⋅f  bf⋅f⎥
        ⎢                                                                                                                                              ⎥
        ⎢aq⋅g  ar⋅g  aq⋅h  ar⋅h  aq⋅i  ar⋅i  as⋅g  at⋅g  as⋅h  at⋅h  as⋅i  at⋅i  au⋅g  av⋅g  au⋅h  av⋅h  au⋅i  av⋅i  aw⋅g  ax⋅g  aw⋅h  ax⋅h  aw⋅i  ax⋅i⎥
        ⎢                                                                                                                                              ⎥
        ⎢ay⋅g  az⋅g  ay⋅h  az⋅h  ay⋅i  az⋅i  ba⋅g  bb⋅g  ba⋅h  bb⋅h  ba⋅i  bb⋅i  bc⋅g  bd⋅g  bc⋅h  bd⋅h  bc⋅i  bd⋅i  be⋅g  bf⋅g  be⋅h  bf⋅h  be⋅i  bf⋅i⎥
        ⎢                                                                                                                                              ⎥
        ⎢a⋅bg  a⋅bh  b⋅bg  b⋅bh  bg⋅c  bh⋅c  a⋅bi  a⋅bj  b⋅bi  b⋅bj  bi⋅c  bj⋅c  a⋅bk  a⋅bl  b⋅bk  b⋅bl  bk⋅c  bl⋅c  a⋅bm  a⋅bn  b⋅bm  b⋅bn  bm⋅c  bn⋅c⎥
        ⎢                                                                                                                                              ⎥
        ⎢a⋅bo  a⋅bp  b⋅bo  b⋅bp  bo⋅c  bp⋅c  a⋅bq  a⋅br  b⋅bq  b⋅br  bq⋅c  br⋅c  a⋅bs  a⋅bt  b⋅bs  b⋅bt  bs⋅c  bt⋅c  a⋅bu  a⋅bv  b⋅bu  b⋅bv  bu⋅c  bv⋅c⎥
        ⎢                                                                                                                                              ⎥
        ⎢bg⋅d  bh⋅d  bg⋅e  bh⋅e  bg⋅f  bh⋅f  bi⋅d  bj⋅d  bi⋅e  bj⋅e  bi⋅f  bj⋅f  bk⋅d  bl⋅d  bk⋅e  bl⋅e  bk⋅f  bl⋅f  bm⋅d  bn⋅d  bm⋅e  bn⋅e  bm⋅f  bn⋅f⎥
        ⎢                                                                                                                                              ⎥
        ⎢bo⋅d  bp⋅d  bo⋅e  bp⋅e  bo⋅f  bp⋅f  bq⋅d  br⋅d  bq⋅e  br⋅e  bq⋅f  br⋅f  bs⋅d  bt⋅d  bs⋅e  bt⋅e  bs⋅f  bt⋅f  bu⋅d  bv⋅d  bu⋅e  bv⋅e  bu⋅f  bv⋅f⎥
        ⎢                                                                                                                                              ⎥
        ⎢bg⋅g  bh⋅g  bg⋅h  bh⋅h  bg⋅i  bh⋅i  bi⋅g  bj⋅g  bi⋅h  bj⋅h  bi⋅i  bj⋅i  bk⋅g  bl⋅g  bk⋅h  bl⋅h  bk⋅i  bl⋅i  bm⋅g  bn⋅g  bm⋅h  bn⋅h  bm⋅i  bn⋅i⎥
        ⎢                                                                                                                                              ⎥
        ⎢bo⋅g  bp⋅g  bo⋅h  bp⋅h  bo⋅i  bp⋅i  bq⋅g  br⋅g  bq⋅h  br⋅h  bq⋅i  br⋅i  bs⋅g  bt⋅g  bs⋅h  bt⋅h  bs⋅i  bt⋅i  bu⋅g  bv⋅g  bu⋅h  bv⋅h  bu⋅i  bv⋅i⎥
        ⎢                                                                                                                                              ⎥
        ⎢a⋅bw  a⋅bx  b⋅bw  b⋅bx  bw⋅c  bx⋅c  a⋅by  a⋅bz  b⋅by  b⋅bz  by⋅c  bz⋅c  a⋅ca  a⋅cb  b⋅ca  b⋅cb  c⋅ca  c⋅cb  a⋅cc  a⋅cd  b⋅cc  b⋅cd  c⋅cc  c⋅cd⎥
        ⎢                                                                                                                                              ⎥
        ⎢a⋅ce  a⋅cf  b⋅ce  b⋅cf  c⋅ce  c⋅cf  a⋅cg  a⋅ch  b⋅cg  b⋅ch  c⋅cg  c⋅ch  a⋅ci  a⋅cj  b⋅ci  b⋅cj  c⋅ci  c⋅cj  a⋅ck  a⋅cl  b⋅ck  b⋅cl  c⋅ck  c⋅cl⎥
        ⎢                                                                                                                                              ⎥
        ⎢bw⋅d  bx⋅d  bw⋅e  bx⋅e  bw⋅f  bx⋅f  by⋅d  bz⋅d  by⋅e  bz⋅e  by⋅f  bz⋅f  ca⋅d  cb⋅d  ca⋅e  cb⋅e  ca⋅f  cb⋅f  cc⋅d  cd⋅d  cc⋅e  cd⋅e  cc⋅f  cd⋅f⎥
        ⎢                                                                                                                                              ⎥
        ⎢ce⋅d  cf⋅d  ce⋅e  cf⋅e  ce⋅f  cf⋅f  cg⋅d  ch⋅d  cg⋅e  ch⋅e  cg⋅f  ch⋅f  ci⋅d  cj⋅d  ci⋅e  cj⋅e  ci⋅f  cj⋅f  ck⋅d  cl⋅d  ck⋅e  cl⋅e  ck⋅f  cl⋅f⎥
        ⎢                                                                                                                                              ⎥
        ⎢bw⋅g  bx⋅g  bw⋅h  bx⋅h  bw⋅i  bx⋅i  by⋅g  bz⋅g  by⋅h  bz⋅h  by⋅i  bz⋅i  ca⋅g  cb⋅g  ca⋅h  cb⋅h  ca⋅i  cb⋅i  cc⋅g  cd⋅g  cc⋅h  cd⋅h  cc⋅i  cd⋅i⎥
        ⎢                                                                                                                                              ⎥
        ⎣ce⋅g  cf⋅g  ce⋅h  cf⋅h  ce⋅i  cf⋅i  cg⋅g  ch⋅g  cg⋅h  ch⋅h  cg⋅i  ch⋅i  ci⋅g  cj⋅g  ci⋅h  cj⋅h  ci⋅i  cj⋅i  ck⋅g  cl⋅g  ck⋅h  cl⋅h  ck⋅i  cl⋅i⎦
    '''
    assert pretty(C, num_columns=10000) == dedent(expected).strip()


def test_inter_product_left_kron():
    # Define symbolic variables
    coms = symmat(5, 'a'), symmat(2, 'b')
    C = kron(*coms)
    # print('C')
    # pprint(C, num_columns=10000)

    E = symmat(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    Z = inter_product(C, E, 10)
    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)
    expected = kron(E, coms[0], coms[1])
    assert Z == expected


def test_inter_product_right_kron():
    # Define symbolic variables
    coms = symmat(5, 'a'), symmat(2, 'b')
    C = kron(*coms)
    # print('C')
    # pprint(C, num_columns=10000)

    E = symmat(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    Z = inter_product(C, E, 1)
    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)

    expected = kron(coms[0], coms[1], E)
    assert Z == expected


def test_inter_product_5_3_2():
    coms = symmat(5, 'a'), symmat(2, 'b')
    C = kron(*coms)
    # print('C')
    # pprint(C, num_columns=10000)

    E = symmat(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    Z = inter_product(C, E, 2)

    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)
    expected = kron(coms[0], E, coms[1])
    assert Z == expected


def test_inter_product_2_3_4():
    coms = symmat(2, 'a'), symmat(2, 'b'), symmat(2, 'c')
    A = kron(*coms)
    # print('A')
    # pprint(A, num_columns=10000)

    E = symmat(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    Z = mesh_product(A, (E,), (4,))
    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)

    expected = kron(coms[0], E, coms[1], coms[2])
    # print('expected')
    # pprint(expected, num_columns=10000)

    assert Z == expected


def test_inter_product_4_3_2():
    coms = symmat(2, 'a'), symmat(2, 'b'), symmat(2, 'c')
    A = kron(*coms)
    # print('A')
    # pprint(A, num_columns=10000)

    E = symmat(3, 'e')
    # print('E')
    # pprint(E)

    # execute
    Z = mesh_product(A, (E,), (2,))
    # print('Z', flush=True)
    # pprint(Z, num_columns=10000)

    expected = kron(coms[0], coms[1], E, coms[2])
    # print('expected')
    # pprint(expected, num_columns=10000)

    assert Z == expected


def test_mesh_product_16_3_2_3_2():
    coms = symmat(2, 'a'), symmat(2, 'b'), symmat(2, 'c')
    A = kron(*coms)
    # print('\nA')
    # pprint(A, num_columns=10000)

    E = symmat(2, 'e')
    # print('\nE')
    # pprint(E)

    F = symmat(2, 'f')
    # print('\nF')
    # pprint(F, num_columns=10000)

    # execute
    Z = mesh_product(A, (E, F), (4, 2))
    # print('\nZ', flush=True)
    # pprint(Z, num_columns=10000)

    expected = kron(coms[0], E, coms[1], F, coms[2])
    # print('\nexpected')
    # pprint(expected, num_columns=10000)

    assert Z == expected
