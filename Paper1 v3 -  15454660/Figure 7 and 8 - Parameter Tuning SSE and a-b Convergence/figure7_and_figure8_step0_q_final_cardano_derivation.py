# Filename: derive_q_final_logic.py
# Purpose: Symbolically analyze the cubic equation aq^3 - bq - L = 0
#          to demonstrate the origin of the solution logic used in
#          WUT's q_final_cardano function.
# Date: May 18, 2025

import sympy

def derive_cubic_solution_logic():
    """
    Uses SymPy to symbolically analyze the cubic equation relevant to q_final.
    """
    print("--- Symbolic Analysis of aq^3 - bq - L = 0 ---")

    # Define symbolic variables
    q = sympy.symbols('q')
    L, a, b = sympy.symbols('L a b', real=True, positive=True) # Assume a, b > 0 for physical potential

    # Define the cubic equation: eq = 0
    # Our equation is a*q**3 - b*q - L = 0
    # SymPy's solve works on expressions, so we define the expression that should be zero.
    cubic_expr = a * q**3 - b * q - L
    print(f"\nCubic Equation: {cubic_expr} = 0")

    # --- 1. Symbolic Solution using SymPy's solver ---
    print("\n--- 1. General Symbolic Solutions (from SymPy) ---")
    try:
        solutions = sympy.solve(cubic_expr, q)
        print("SymPy found the following symbolic solutions for q:")
        for i, sol in enumerate(solutions):
            print(f"  Solution {i+1}:")
            sympy.pprint(sol, use_unicode=True)
            print("-" * 20)
        print("\nNote: These are the general complex roots. Cardano's method and the")
        print("trigonometric form for real roots are derived from these general forms")
        print("under specific conditions on the discriminant.")
    except Exception as e:
        print(f"Error solving symbolically with SymPy: {e}")
        print("This might happen if the equation is too complex for direct general solution display.")

    # --- 2. Analysis via Standard Depressed Cubic: x^3 + px + r_ = 0 ---
    # To transform aq^3 - bq - L = 0  to x^3 + p_norm*x + r_norm = 0, let q = x.
    # Divide by a: q^3 - (b/a)q - (L/a) = 0
    # So, p_norm = -b/a
    # And r_norm = -L/a
    p_norm = -b / a
    r_norm = -L / a
    print("\n--- 2. Analysis via Depressed Cubic Form (q^3 + p_norm*q + r_norm = 0) ---")
    print(f"Normalized p_norm = -b/a:")
    sympy.pprint(p_norm)
    print(f"Normalized r_norm = -L/a:")
    sympy.pprint(r_norm)

    # --- 3. Discriminant Analysis ---
    # For x^3 + px + r_ = 0, the discriminant is Delta = (r_/2)^2 + (p/3)^3
    discriminant_norm = (r_norm / 2)**2 + (p_norm / 3)**3
    print("\n--- 3. Discriminant (for normalized form q^3 + p_norm*q + r_norm = 0) ---")
    print("Discriminant Delta_norm = (r_norm/2)^2 + (p_norm/3)^3:")
    sympy.pprint(sympy.simplify(discriminant_norm))

    print("\nConditions for real roots based on Delta_norm:")
    print("  - If Delta_norm > 0: One real root, two complex conjugate roots.")
    print("  - If Delta_norm = 0: Three real roots, at least two are equal.")
    print("                       (If p_norm=0 and r_norm=0, then q=0 is a triple root).")
    print("                       (If p_norm!=0, then one double root and one single root).")
    print("  - If Delta_norm < 0: Three distinct real roots.")

    # Substitute back p_norm and r_norm in terms of L, a, b
    discriminant_original_vars = (( -L / (2*a) )**2 + ( -b / (3*a) )**3)
    print("\nDiscriminant Delta in terms of original L, a, b:")
    sympy.pprint(sympy.simplify(discriminant_original_vars))
    # This simplifies to: (27*L**2*a - 4*b**3) / (108*a**3)
    # The sign is determined by the numerator: 27*L**2*a - 4*b**3
    # Condition for 3 real roots (Delta < 0): 27*L**2*a - 4*b**3 < 0
    # => 27*L**2*a < 4*b**3
    # => L**2 < (4*b**3) / (27*a)
    # => |L| < sqrt((4*b**3) / (27*a)) = (2*b*sqrt(b)) / (3*sqrt(3*a))
    # This is L_crit, critical L.
    L_crit_sq = (4 * b**3) / (27 * a)
    print(f"\nCondition for three distinct real roots (Delta < 0) is L^2 < L_crit^2, where L_crit^2 =")
    sympy.pprint(L_crit_sq)
    print(f"This means |L| < L_crit = (2b*sqrt(b)) / (3*sqrt(3a))")

    # --- 4. Cardano's Formula (for Delta_norm >= 0, one real root case) ---
    print("\n--- 4. Cardano's Formula (One Real Root when Delta_norm >= 0) ---")
    # x = cbrt(-r_norm/2 + sqrt(Delta_norm)) + cbrt(-r_norm/2 - sqrt(Delta_norm))
    A_term_cardano = sympy.cbrt(-r_norm/2 + sympy.sqrt(discriminant_norm))
    B_term_cardano = sympy.cbrt(-r_norm/2 - sympy.sqrt(discriminant_norm))
    q_cardano_real_root = A_term_cardano + B_term_cardano
    print("One real root (q) from Cardano's formula:")
    sympy.pprint(q_cardano_real_root)

    # --- 5. Trigonometric Solution (for Delta_norm < 0, three distinct real roots) ---
    print("\n--- 5. Trigonometric Solution (Three Distinct Real Roots when Delta_norm < 0) ---")
    # Requires p_norm < 0 (i.e., b/a > 0, which is true since a,b > 0)
    # Roots are: q_k = 2 * sqrt(-p_norm/3) * cos( (1/3)*acos(-r_norm / (2*sqrt(-(p_norm/3)^3))) + 2*pi*k/3 ) for k=0,1,2
    prefactor_trig = 2 * sympy.sqrt(-p_norm / 3)
    # Need to ensure -p_norm/3 is positive: -(-b/a)/3 = b/(3a) > 0. OK.
    cos_arg_num_trig = -r_norm / 2
    cos_arg_den_trig = sympy.sqrt((-p_norm / 3)**3)
    # Need to ensure (-p_norm/3)^3 is positive: (b/(3a))^3 > 0. OK.
    phi_trig = sympy.acos(cos_arg_num_trig / cos_arg_den_trig) # This is (1/3) * arccos(...)
    
    print(f"Let prefactor = 2 * sqrt(-p_norm/3) = 2 * sqrt(b/(3a))")
    print(f"Let phi_angle = (1/3) * acos( -r_norm / (2 * sqrt((-p_norm/3)^3)) )")
    print(f"   phi_angle = (1/3) * acos( (L/2a) / sqrt((b/3a)^3) )")

    q_trig_k0 = prefactor_trig * sympy.cos(phi_trig / 3)
    q_trig_k1 = prefactor_trig * sympy.cos(phi_trig / 3 + 2 * sympy.pi / 3)
    q_trig_k2 = prefactor_trig * sympy.cos(phi_trig / 3 + 4 * sympy.pi / 3)

    print("\nThe three distinct real roots (q_k) are:")
    print("q_k0 (typically corresponds to sgn(L) for L > 0, or largest magnitude for L=L_crit):")
    sympy.pprint(q_trig_k0)
    print("\nq_k1 (typically corresponds to sgn(L) for L < 0, or other sign for L=L_crit):")
    sympy.pprint(q_trig_k1)
    print("\nq_k2 (the remaining root):")
    sympy.pprint(q_trig_k2)

    # --- 6. Logic for Selecting q_final (Physical Stable Root) ---
    print("\n--- 6. Logic for Selecting q_final (The Physical Stable Root) ---")
    print("The WUT mechanistic model (Appendix D of the paper) shows that the detector")
    print("state q relaxes to the stable fixed point that minimizes the effective potential")
    print("U_eff = a*q^4/4 - b*q^2/2 - L*q.")
    print("This analysis proves that sgn(q_final) = sgn(L) for L != 0.")
    print("When L=0, q_final=0 is chosen as the continuation from small L, although")
    print("the stable states are +/-sqrt(b/a); the *change* is driven by L.")

    print("\nTherefore, the implementation of `q_final_cardano(L, a, b)` must select:")
    print("  - The single real root (from Cardano) if Delta_norm > 0.")
    print("  - The appropriate root from the trigonometric solution if Delta_norm < 0, such that:")
    print("    - If L > 0, q_final is positive (typically q_trig_k0).")
    print("    - If L < 0, q_final is negative (typically q_trig_k1, or q_trig_k2 if k1 is positive).")
    print("      (The standard selection is k=0 for L>0, and k=1 (phi/3 + 2pi/3) for L<0 to get the root that tends to -sqrt(b/a)).")
    print("  - Handles the Delta_norm = 0 case correctly (boundary roots).")
    print("  - Returns 0 if L = 0.")

    print("\nThe Python function `q_final_cardano` in the numerical scripts implements this logic,")
    print("using numerical comparisons and robust handling of floating-point arithmetic for")
    print("the different cases determined by L relative to L_crit (which depends on the discriminant).")

if __name__ == '__main__':
    derive_cubic_solution_logic()