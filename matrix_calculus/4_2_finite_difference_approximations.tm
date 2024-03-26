<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Lecture 4 Part 2: Finite-difference
  Approximations>|<\doc-subtitle>
    MIT 18.S096 Matrix Calculus For Machine Learning and Beyond
  </doc-subtitle>|<doc-date|March 25, 2024>>

  <section|Finite-difference approximation for vector-to-scalar functions>

  This part largely follows from Section 8.1 of Numerical Optimization by
  Nocedal and Wright.

  There are so many assumptions (colored in red)!

  <subsection|Truncation error>

  Consider a twice continuously differentiable function
  <math|f:\<bbb-R\><rsup|n>\<rightarrow\>\<bbb-R\>>. Let
  <math|x,p\<in\>\<bbb-R\><rsup|n>>. Then, by Taylor's theorem,

  <\equation*>
    f<around*|(|x+p|)>=f<around*|(|x|)>+\<nabla\>f<around*|(|x|)><rsup|T>p+<frac|1|2>p<rsup|T>\<nabla\><rsup|2>f<around*|(|x+t
    p|)>p<space|1em><text|for some <math|t\<in\><around*|(|0,1|)>>>.
  </equation*>

  Note that this is actually pretty interesting because
  <math|\<nabla\><rsup|2>f<around*|(|x+t p|)>> is the Hessian of a point on
  the line that interpolates <math|x> and <math|p>.\ 

  Continuing on:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|x+p|)>>|<cell|=>|<cell|f<around*|(|x|)>+\<nabla\>f<around*|(|x|)><rsup|T>p+<frac|1|2>p<rsup|T>\<nabla\><rsup|2>f<around*|(|x+t
    p|)>p>>|<row|<cell|f<around*|(|x+p|)>-f<around*|(|x|)>-\<nabla\>f<around*|(|x|)><rsup|T>p>|<cell|=>|<cell|<frac|1|2>p<rsup|T>\<nabla\><rsup|2>f<around*|(|x+t
    p|)>p>>|<row|<cell|<around*|\<\|\|\>|f<around*|(|x+p|)>-f<around*|(|x|)>-\<nabla\>f<around*|(|x|)><rsup|T>p|\<\|\|\>>>|<cell|=>|<cell|<around*|\<\|\|\>|<frac|1|2>p<rsup|T>\<nabla\><rsup|2>f<around*|(|x+t
    p|)>p|\<\|\|\>>>>|<row|<cell|<around*|\<\|\|\>|f<around*|(|x+p|)>-f<around*|(|x|)>-\<nabla\>f<around*|(|x|)><rsup|T>p|\<\|\|\>>>|<cell|\<leq\>>|<cell|<frac|1|2><around*|\<\|\|\>|p<rsup|T>|\<\|\|\>>
    <around*|\<\|\|\>|\<nabla\><rsup|2>f<around*|(|x+t p|)>|\<\|\|\>>
    <around*|\<\|\|\>|p|\<\|\|\>>.>>>>
  </eqnarray*>

  <with|color|red|Let <math|L> be the bound on
  <math|<around*|\<\|\|\>|\<nabla\><rsup|2>f<around*|(|\<cdot\>|)>|\<\|\|\>>>
  in the region of interest.> Obtain

  <\eqnarray*>
    <tformat|<table|<row|<cell|<around*|\<\|\|\>|f<around*|(|x+p|)>-f<around*|(|x|)>-\<nabla\>f<around*|(|x|)><rsup|T>p|\<\|\|\>>>|<cell|\<leq\>>|<cell|<around*|(|L/2|)><around*|\<\|\|\>|p|\<\|\|\>><rsup|2>.>>>>
  </eqnarray*>

  If we choose <math|p> to be <math|\<varepsilon\>e<rsub|i>>, then
  <math|\<nabla\>f<around*|(|x|)><rsup|T>p=\<varepsilon\><around*|(|\<partial\>f/\<partial\>x<rsub|i>|)>>
  and <math|<around*|\<\|\|\>|p|\<\|\|\>><rsup|2>=\<varepsilon\><rsup|2>>.
  Obtain

  <\equation*>
    -<around*|(|L/2|)>\<varepsilon\><rsup|2>\<leq\>f<around*|(|x+\<varepsilon\>e<rsub|i>|)>-f<around*|(|x|)>-\<varepsilon\><frac|\<partial\>f|\<partial\>x<rsub|i>><around*|(|x|)>\<leq\><around*|(|L/2|)>\<varepsilon\><rsup|2>
  </equation*>

  <\equation*>
    -f<around*|(|x+\<varepsilon\>e<rsub|i>|)>+f<around*|(|x|)>-<around*|(|L/2|)>\<varepsilon\><rsup|2>\<leq\>-\<varepsilon\><frac|\<partial\>f|\<partial\>x<rsub|i>><around*|(|x|)>\<leq\>-f<around*|(|x+\<varepsilon\>e<rsub|i>|)>+f<around*|(|x|)>+<around*|(|L/2|)>\<varepsilon\><rsup|2>
  </equation*>

  <\equation*>
    <frac|f<around*|(|x+\<varepsilon\>e<rsub|i>|)>-f<around*|(|x|)>|\<varepsilon\>>+<around*|(|L/2|)>\<varepsilon\>\<geq\><frac|\<partial\>f|\<partial\>x<rsub|i>><around*|(|x|)>\<geq\><frac|f<around*|(|x+\<varepsilon\>e<rsub|i>|)>-f<around*|(|x|)>|\<varepsilon\>>-<around*|(|L/2|)>\<varepsilon\>
  </equation*>

  <\equation*>
    <frac|\<partial\>f|\<partial\>x<rsub|i>><around*|(|x|)>=<frac|f<around*|(|x+\<varepsilon\>e<rsub|i>|)>-f<around*|(|x|)>|\<varepsilon\>>+\<delta\><rsub|\<varepsilon\>><space|1em>where<space|1em><around*|\||\<delta\><rsub|\<varepsilon\>>|\|>\<leq\><around*|(|L/2|)>\<varepsilon\>.
  </equation*>

  <math|\<delta\><rsub|\<varepsilon\>>> is commonly referred to as the
  <with|font-shape|italic|truncation> error. This becomes
  <with|font-shape|italic|forward difference> formula if we ignore the
  <math|\<delta\><rsub|\<varepsilon\>>> term, which becomes smaller and
  smaller as <math|\<varepsilon\>\<rightarrow\>0>.\ 

  <subsection|Round-off error>

  For simplicity, <with|color|red|assume that the relative error in the
  computed <math|f> is bounded by <math|<with|font-series|bold|u>>>
  (<math|<with|font-series|bold|u>> is about <math|10<rsup|-16>> in
  double-precision representation) (I don't really know when this assumption
  is reasonable.):

  <\eqnarray*>
    <tformat|<table|<row|<cell|<around*|\||comp<around*|(|f<around*|(|x|)>-f<around*|(|x|)>|)>|\|>>|<cell|\<leq\>>|<cell|<with|font-series|bold|u>L<rsub|f>>>|<row|<cell|<around*|\||comp<around*|(|f<around*|(|x+\<varepsilon\>e<rsub|i>|)>-f<around*|(|x+\<varepsilon\>e<rsub|i>|)>|)>|\|>>|<cell|\<leq\>>|<cell|<with|font-series|bold|u>L<rsub|f>,>>>>
  </eqnarray*>

  where <math|comp<around*|(|\<cdot\>|)>> denotes the computed value, and
  <with|color|red|<math|L<rsub|f>> is a bound on the value of
  <math|<around*|\||f<around*|(|\<cdot\>|)>|\|>> in the region of interest>.
  If we use the computed values in the forward difference formula

  <\equation*>
    <frac|\<partial\>f|\<partial\>x<rsub|i>><around*|(|x|)>=<frac|f<around*|(|x+\<varepsilon\>e<rsub|i>|)>-f<around*|(|x|)>|\<varepsilon\>>+\<delta\><rsub|\<varepsilon\>>,
  </equation*>

  we get

  <\equation*>
    <around*|\||\<delta\><rsub|\<varepsilon\>>|\|>\<leq\><wide*|<around*|(|L/2|)>\<varepsilon\>|\<wide-underbrace\>><rsub|<text|truncation
    error>>+<wide*|2<with|font-series|bold|u>L<rsub|f>/\<varepsilon\>|\<wide-underbrace\>><rsub|<text|round-off
    error>>.
  </equation*>

  Notice how the truncation error decreases as
  <math|\<varepsilon\>\<rightarrow\>0>, while the round-off error blows up as
  <math|\<varepsilon\>\<rightarrow\>0>.

  Taking the derivaive of this expression with respect to
  <math|\<varepsilon\>> and setting it to zero, we obtain

  <\equation*>
    \<varepsilon\><rsup|\<ast\>>=<sqrt|<frac|4L<rsub|f>
    <with|font-series|bold|u>|L>>.
  </equation*>

  <with|color|red|Assuming that <math|4L<rsub|f>/L\<approx\>1>>, we get

  <\equation*>
    \<varepsilon\><rsup|\<ast\>>=<sqrt|<with|font-series|bold|u>>,
  </equation*>

  which is what's used in most packages. In PyTorch,
  <math|<with|font-series|bold|u>=1\<times\>10<rsup|-6>>.\ 

  <section|Finite-difference approximation for matrix-to-matrix functions>

  Let <math|v> be a generic vector (could be a scalar, a vector or a matrix).
  Then, in the differential notation, we have

  <\equation*>
    f<around*|(|v+d v|)>-f<around*|(|v|)>=f<rprime|'><around*|(|v|)><around*|[|d
    v|]>+<text|higher-order terms>.
  </equation*>

  When <math|d v> is very small, we have

  <\equation*>
    f<around*|(|v+d v|)>-f<around*|(|v|)>\<approx\>f<rprime|'><around*|(|v|)><around*|[|d
    v|]>.
  </equation*>

  <math|f<rprime|'><around*|(|v|)><around*|[|\<cdot\>|]>> would typically be
  something that we derive by hand or autograd \U we can check the
  correctness of this derivation by first choosing a small <math|d v> and
  then comparing <math|f<around*|(|v+d v|)>-f<around*|(|v|)>> and
  <math|f<rprime|'><around*|(|v|)><around*|[|d v|]>> \U their difference
  should be small.

  <with|font-shape|italic|How should we measure their difference?> Ratio of
  norms:

  <\equation*>
    <frac|<around*|\<\|\|\>|estimated-truth|\<\|\|\>>|<around*|\<\|\|\>|truth|\<\|\|\>>>
  </equation*>

  For matrices, we would use the Frobenius norm.

  <math|f<around*|(|v+d v|)>-f<around*|(|v|)>> is called the
  <with|font-shape|italic|forward> difference. We also could have chosen
  <math|f<around*|(|v|)>-f<around*|(|v-d v|)>>, the
  <with|font-shape|italic|backward> difference. But it turns out that the
  <with|font-shape|italic|central> difference usually works the best:

  <\equation*>
    f<around*|(|v+<frac|1|2>d v|)>-f<around*|(|v-<frac|1|2>d v|)>.
  </equation*>

  \;

  \;

  \;

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|1.1|1>>
    <associate|auto-3|<tuple|1.2|2>>
    <associate|auto-4|<tuple|2|2>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Finite-difference
      for vector-to-scalar functions> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Truncation error
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|1.2<space|2spc>Round-off error
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Finite-difference
      for matrix-to-matrix functions> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>