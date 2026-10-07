"""Run:  .venv\Scripts\python.exe tests\phi_misses.py
Shows, for sample questions, what the identifier scrubber removed and which names it MISSED (a miss = a name still present after scrubbing)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phi

# (question, names that must NOT remain after scrubbing)
CASES = [
    ("John has seizures", ["John"]),
    ("my patient john has fever and cough", ["john"]),
    ("Mary Smith, a 5-year-old girl with autism, was seen by Dr. Patel", ["Mary", "Smith", "Patel"]),
    ("patient Rahul Kumar 7 year old on risperidone 0.5 mg", ["Rahul", "Kumar"]),
    ("Ayesha, 6 years old, non-verbal. How to treat autism?", ["Ayesha"]),
    ("Mrs. Sharma's MRI shows lesions, what treatment options exist for MS?", ["Sharma"]),
    ("The child is named Arjun Mehta and has ADHD", ["Arjun", "Mehta"]),
    ("Muhammad Ali, 8, has epilepsy: DOB 12/03/2016, phone +91 98765 43210, MRN 4482913", ["Muhammad", "Ali"]),
    ("Priya's HbA1c is 8.2% and she is on metformin 500 mg, how to adjust", ["Priya"]),
    ("my son Arjun has autism what therapy helps", ["Arjun"]),
    ("li wei, 9 year old, asthma, uses salbutamol", ["li wei", "Li Wei"]),
    ("ritu has asthma, what inhaler", ["ritu", "Ritu"]),
    ("john smith has seizures", ["john", "smith"]),
    ("Baby Anaya, 3 months old, fever 39.2 C", ["Anaya"]),
    ("Email dr.rao@hospital.org or call 044-2345-6789 about the 5-year-old", []),
    ("lives at 12 Gandhi Road, Chennai 600004; patient ID 99231-A", []),
]
# clinical questions that contain person-like words and must come through UNCHANGED
SAFE = ["Parkinson's disease treatment in elderly", "Down syndrome and hearing loss", "What is the dose for Wilson disease", "Crohn's disease in children",
        "Guillain-Barre syndrome IVIG dosing", "5-year-old male on Risperdal 0.5 mg with HbA1c 7.2%", "Alzheimer disease new drugs", "Tourette syndrome treatment",
        "Hodgkin lymphoma in adolescents", "Glasgow Coma Scale 8 management"]

if __name__ == "__main__":
    phi.warm_up()
    missed = 0
    for q, names in CASES:
        r = phi.scrub(q)
        left = [n for n in names if n.lower() in r.text.lower()]
        missed += bool(left)
        print(("MISS " if left else "ok   ") + f"{q!r}\n        -> {r.text!r}  removed={r.removed}" + (f"  STILL PRESENT: {left}" if left else ""))
    print()
    changed = [q for q in SAFE if phi.scrub(q).text != q]
    for q in SAFE:
        r = phi.scrub(q)
        print(("CHANGED " if r.text != q else "intact  ") + f"{q!r}" + (f" -> {r.text!r}" if r.text != q else ""))
    print(f"\nnames missed in {missed} of {len(CASES)} cases; clinical questions altered: {len(changed)} of {len(SAFE)}")
