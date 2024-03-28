<TeXmacs|2.1.1>

<style|<tuple|generic|framed-theorems|number-europe>>

<\body>
  <doc-data|<doc-title|Chapter 8: Calculating
  Derivatives>|<doc-subtitle|Comments and Proofs>>

  <\table-of-contents|toc>
    <vspace*|1fn><with|font-series|bold|math-font-series|bold|1<space|2spc>Finite-difference
    derivative approximations> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-1><vspace|0.5fn>

    <with|par-left|1tab|1.1<space|2spc>Approximating the gradient
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-2>>

    <with|par-left|2tab|1.1.1<space|2spc>Forward-difference approximation
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-3>>

    <with|par-left|2tab|1.1.2<space|2spc>Central-difference approximation
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-4>>
  </table-of-contents>

  <section|Finite-difference derivative approximations>

  <subsection|Approximating the gradient>

  <subsubsection|Forward-difference approximation>

  \;

  <subsubsection|Central-difference approximation>

  <\theorem>
    (Error of central-difference approximation)

    Suppose <math|f:\<bbb-R\><rsup|n>\<rightarrow\>\<bbb-R\>> is twice
    continuously differentiable and its second (partial) derivatives are
    Lipschitz continuous. Then

    <\equation*>
      <frac|\<partial\>f|\<partial\>x<rsub|i>><around*|(|x|)>=<frac|f<around*|(|x+\<varepsilon\>e<rsub|i>|)>-f<around*|(|x-\<varepsilon\>e<rsub|i>|)>|2\<varepsilon\>>+O<around*|(|\<varepsilon\><rsup|2>|)>.
    </equation*>
  </theorem>

  <\remark>
    Lipschitz continuity implies continuity, but not vice versa.
  </remark>

  <\proof>
    Let <math|x,p\<in\>\<bbb-R\><rsup|n>>. By Taylor's theorem, we have that\ 

    <\equation*>
      f<around*|(|x+p|)>=f<around*|(|x|)>+\<nabla\>f<around*|(|x|)><rsup|T>p+<frac|1|2>p<rsup|T>\<nabla\><rsup|2>f<around*|(|x+t
      p|)>p
    </equation*>

    for some <math|t\<in\><around*|(|0,1|)>>.\ 

    Since the second derivatives are Lipschitz continuous, we can find
    <math|L\<gtr\>0> such that

    <\equation*>
      <around*|\||<frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x+t
      p|)>-<frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x|)>|\|>=L<around*|\<\|\|\>|x+t
      p-x|\<\|\|\>>=t L<around*|\<\|\|\>|p|\<\|\|\>>.
    </equation*>

    For the definition of Lipschitz continuity for multivariate functions,
    see <with|color|#a0a0a0|Appendix A.2 Continuity and Limits> (page 623).

    Converting the absolute value on the left to an inequality, obtain

    <\eqnarray*>
      <tformat|<table|<row|<cell|>|<cell|>|<cell|-t
      L<around*|\<\|\|\>|p|\<\|\|\>>\<leq\><frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x+t
      p|)>-<frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x|)>\<leq\>t
      L<around*|\<\|\|\>|p|\<\|\|\>>>>|<row|<cell|>|<cell|\<Rightarrow\>>|<cell|-t
      L<around*|\<\|\|\>|p|\<\|\|\>>\<leq\><frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x+t
      p|)>-<frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x|)>\<leq\>t
      L<around*|\<\|\|\>|p|\<\|\|\>>>>|<row|<cell|>|<cell|\<Rightarrow\>>|<cell|<frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x|)>-t
      L<around*|\<\|\|\>|p|\<\|\|\>>\<leq\><frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x+t
      p|)>\<leq\>t L<around*|\<\|\|\>|p|\<\|\|\>>+<frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x|)>>>|<row|<cell|>|<cell|\<Rightarrow\>>|<cell|<frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x+t
      p|)>=<frac|\<partial\>f|\<partial\>x<rsub|i>\<partial\>x<rsub|j>><around*|(|x|)>+O<around*|(|<around*|\<\|\|\>|p|\<\|\|\>>|)>.>>>>
    </eqnarray*>

    Substituting this result into Taylor's theorem, obtain

    <\eqnarray*>
      <tformat|<table|<row|<cell|f<around*|(|x+p|)>>|<cell|=>|<cell|f<around*|(|x|)>+\<nabla\>f<around*|(|x|)><rsup|T>p+<frac|1|2>p<rsup|T><around*|<left|(|2>|\<nabla\><rsup|2>f<around*|(|x|)>+O<around*|(|<around*|\<\|\|\>|p|\<\|\|\>>|)>|<right|)|2>>p>>|<row|<cell|>|<cell|=>|<cell|f<around*|(|x|)>+\<nabla\>f<around*|(|x|)><rsup|T>p+<frac|1|2>p<rsup|T>\<nabla\><rsup|2>f<around*|(|x|)>p+O<around*|(|<around*|\<\|\|\>|p|\<\|\|\>><rsup|3>|)>.>>>>
    </eqnarray*>

    Similarly,\ 

    <\equation*>
      f<around*|(|x-p|)>=f<around*|(|x|)>-\<nabla\>f<around*|(|x|)><rsup|T>p+<frac|1|2>p<rsup|T>\<nabla\><rsup|2>f<around*|(|x|)>p+O<around*|(|<around*|\<\|\|\>|p|\<\|\|\>><rsup|3>|)>.
    </equation*>

    For the remainder of the proof, just refer to the book.

    \;
  </proof>

  <section|Automatic Differentiation>

  <section*|Exercises>

  <subsection*|Problem 8.1>

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
    <associate|auto-3|<tuple|1.1.1|1>>
    <associate|auto-4|<tuple|1.1.2|1>>
    <associate|auto-5|<tuple|2|?>>
    <associate|auto-6|<tuple|2|?>>
    <associate|auto-7|<tuple|2|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Finite-difference
      derivative approximations> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Approximating the gradient
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|2tab>|1.1.1<space|2spc>Forward-difference
      approximation <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|2tab>|1.1.2<space|2spc>Central-difference
      approximation <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Automatic
      Differentiation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Exercises>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <with|par-left|<quote|1tab>|Problem 8.1
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>
    </associate>
  </collection>
</auxiliary>