from datalog_types import *
import pprint as pp


# --- Variables ---
varu  = Var("u")
varx  = Var("x")
vary = Var("y")
varz  = Var("z")
varl  = Var("l")
varr  = Var("r")
varpx = Var("px")
varpy = Var("py")
varcx = Var("cx")
varcy = Var("cy")



# --- Rules (as a Python list) ---
PROGRAM_A = [
    # Seed(U) :- Active(U).
    Rule( head=atom("Seed", varu),  body=[atom("Active", varu)]  ),
    # Seed(U) :- Type(U, "good").
    Rule( head=atom("Seed", varu), body=[atom("Type", varu, "good")]   ),
    # LReach(X,Z,L) :- Edge(X,Z,L), Seed(X), Seed(Z).
    Rule( head=atom("LReach", varx, varz, varl), body=[atom("Edge", varx, varz, varl), atom("Seed", varx), atom("Seed", varz)] ),
    # LReach(X,Z,L) :- Edge(X,Y,L), Seed(Y), LReach(Y,Z,L).
    Rule( head=atom("LReach", varx, varz, varl), body=[atom("Edge", varx, vary, varl), 
                                                        atom("Seed", vary), atom("LReach", vary, varz, varl)] ),
    # Q(x,y) :- LReach(x,y,"db").
    Rule( head=atom("Q", varx, vary), body=[atom("LReach", varx, vary, "db")] ),
]


PROGRAM_B = [
    # SameDepthOpp(X,Y,R) :- Parent(X,R), Parent(Y,R), Color(X,"red"), Color(Y,"blue").
    Rule(
        head=atom("SameDepthOpp", varx, vary, varr),
        body=[
            atom("Parent", varx, varr),
            atom("Parent", vary, varr),
            atom("Color",  varx, "red"),
            atom("Color",  vary, "blue"),
        ],
    ),
    # SameDepthOpp(X,Y,R) :- Parent(X,R), Parent(Y,R), Color(X,"yellow"), Color(Y,"red").
    Rule(
        head=atom("SameDepthOpp", varx, vary, varr),
        body=[
            atom("Parent", varx, varr),
            atom("Parent", vary, varr),
            atom("Color",  varx, "yellow"),
            atom("Color",  vary, "red"),
        ],
    ),
    # SameDepthOpp(X,Y,R) :-
    #   Parent(X,PX), Parent(Y,PY), SameDepthOpp(PX,PY,R),
    #   Color(X,Cx), Color(Y,Cy), DiffColor(Cx,Cy).
    Rule(
        head=atom("SameDepthOpp", varx, vary, varr),
        body=[
            atom("Parent",      varx, varpx),
            atom("Parent",      vary, varpy),
            atom("SameDepthOpp", varpx, varpy, varr),
            atom("Color",       varx, varcx),
            atom("Color",       vary, varcy),
            atom("DiffColor",   varcx, varcy),
        ],
    ),
]

if __name__ == "__main__":
    pp.pprint(PROGRAM_A)
    pp.pprint(PROGRAM_B)
