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

### An alternate-proof from GPT-5

I asked GPT-5 to prove this inequality, and it came up with a similar idea, expressed differently using complex numbers.  Here's the model's proof:

Let `$\omega = e^{i\pi/3} = \cos(\pi/3) + i\sin(\pi/3)$`. Using `$|x - y e^{i\theta}|^2 = x^2 - 2xy\cos\theta + y^2$`, we get
`$$
a^{2}-ab+b^{2} = |\,a - b\omega\,|^{2},\qquad
b^{2}-bc+c^{2} = |\,b\omega - c\omega^{2}\,|^{2},
$$`
and since `$\cos(2\pi/3)=-\tfrac12$`,
`$$
a^{2}+ac+c^{2} = |\,a - c\omega^{2}\,|^{2}.
$$`
Hence the inequality is equivalent to
`$$
|a-b\omega| + |b\omega - c\omega^{2}| \ge |a - c\omega^{2}|,
$$`
which is exactly the triangle inequality for the complex numbers
`$A=a$`, `$B=b\omega$,` and `$C=c\omega^{2}$`.

Equality holds iff `$A,B,C$` are collinear and `$B$` lies on the segment `$AC$`.

### Interactive visualization

Finally, I asked GPT-5 to make an interactive app to visualize the inequality.  The model is able to create a pretty cool visualization!

<iframe
  src="/interactive/triangle-inequality.html"
  width="100%"
  height="1000"
  style="border: 1px solid rgba(148,163,184,0.25); border-radius: 12px;"
  loading="lazy"
></iframe>

Click [here](/interactive/triangle-inequality.html) for a fullscreen version.
