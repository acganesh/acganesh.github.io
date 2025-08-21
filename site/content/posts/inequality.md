---
layout: post
title: Using geometry to prove an inequality
date: 2025-08-20
math: true
description: A fun inequality with a geometric interpretation.
---

### The inequality

I really like this inequality, which I encountered in a past life when I used to do math competitions.  The problem statement is as follows:

*Problem.* Prove that for all `$a,b,c>0$` we have
`$$
\sqrt{a^2 - ab + b^2} + \sqrt{b^2 - bc + c^2} \ge \sqrt{a^2 + ac + c^2}.
$$`

*Proof.* Upon first glance this seems like a problem that requires a bunch of algebra to prove.   But it turns out you can solve it by drawing a picture:

![Draw a picture](/img/inequality.png)

In convex quadrilateral `$ABCD$` above, let `$BA=a$`, `$BD=b$`, and `$BC=c$` where `$\angle ABD=\angle CBD=60^\circ$`. By the Law of Cosines,
`$$
\begin{aligned}
AD &= \sqrt{a^2-2ab\cos 60^\circ+b^2} = \sqrt{a^2-ab+b^2},\\
DC &= \sqrt{b^2-2bc\cos 60^\circ+c^2} = \sqrt{b^2-bc+c^2},\\
AC &= \sqrt{a^2-2ac\cos 120^\circ+c^2} = \sqrt{a^2+ac+c^2}.
\end{aligned}
$$`

By the triangle inequality, we have `$AD+DC\ge AC$`, from which the conclusion follows.

### An algebraic proof from GPT-5

I asked GPT-5 to prove this inequality algebraically with no geometric argument.  It came up with the following proof:

Set
`$$
A=a^2-ab+b^2,\qquad B=b^2-bc+c^2,\qquad C=a^2+ac+c^2.
$$`
We want `$\sqrt A+\sqrt B\ge \sqrt C$`.
Since both sides are nonnegative, it suffices to square once:
`$$
A+B+2\sqrt{AB}\ \ge\ C
\iff
2\sqrt{AB}\ \ge\ C-(A+B).
$$`
A direct computation gives
`$$
C-(A+B)=ab+bc+ac-2b^2.
$$`
Therefore it remains to prove
`$$
2\sqrt{AB}\ \ge\ ab+bc+ac-2b^2.
$$`
Observe that
`$$
\begin{aligned}
\bigl(2\sqrt{AB}\bigr)^2-\bigl(ab+bc+ac-2b^2\bigr)^2
&=4AB-\bigl(ab+bc+ac-2b^2\bigr)^2\\[2mm]
&=3\,(ab-ac+bc)^2\ \ge\ 0.
\end{aligned}
$$`
Hence `$2\sqrt{AB}\ge \bigl|ab+bc+ac-2b^2\bigr|\ge ab+bc+ac-2b^2$`, which implies
`$A+B+2\sqrt{AB}\ge C$`. Consequently,
`$$
\sqrt{a^2-ab+b^2}+\sqrt{b^2-bc+c^2}\ge \sqrt{a^2+ac+c^2}.
$$`
Equality holds iff `$ab-ac+bc=0$`, i.e. `$b=\dfrac{ac}{a+c}$`.


### Interactive visualization

Finally, I asked GPT-5 to make an interactive app to visualize the inequality.  The model is able to create a pretty cool visualization!

<iframe
  src="/interactive/triangle-inequality.html"
  width="100%"
  height="700"
  style="border: 1px solid rgba(148,163,184,0.25); border-radius: 12px;"
  loading="lazy"
></iframe>

Click [here](/interactive/triangle-inequality.html) for a fullscreen version.

Thanks to [Andy Chen](https://asjchen.github.io/) for reviewing a draft of this post.