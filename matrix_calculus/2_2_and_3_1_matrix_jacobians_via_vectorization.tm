<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Lecture 2 Part 2: Vectorization of Matrix Functions +
  Lecture 3 Part 1: Kronecker Products and Jacobians>|<\doc-subtitle>
    MIT 18.S096 Matrix Calculus For Machine Learning and Beyond
  </doc-subtitle>|<doc-date|March 17, 2024>>

  <section|Differential>

  <\equation*>
    f<around*|(|A+d A|)>=f<around*|(|A|)>+f<rprime|'><around*|(|A|)><around*|[|d
    A|]>+<text|higher-order stuff>
  </equation*>

  <\equation*>
    d f<around*|(|A;d A|)>=f<rprime|'><around*|(|A|)><around*|[|d A|]>
  </equation*>

  <section|Example>

  Let <math|f> be the following function that maps from
  <math|\<bbb-R\><rsup|n\<times\>n>> to <math|\<bbb-R\><rsup|n\<times\>n>>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|A|)>>|<cell|=>|<cell|A<rsup|2>>>>>
  </eqnarray*>

  Deriving the difference:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|A+d
    A|)>-f<around*|(|A|)>>|<cell|=>|<cell|<around*|(|A+d
    A|)><rsup|2>-A<rsup|2>>>|<row|<cell|>|<cell|=>|<cell|A<rsup|2>+A
    <around*|(|d A|)>+<around*|(|d A|)> A+<around*|(|d
    A|)><rsup|2>-A<rsup|2>>>|<row|<cell|>|<cell|=>|<cell|A <around*|(|d
    A|)>+<around*|(|d A|)> A+<around*|(|d A|)><rsup|2>>>>>
  </eqnarray*>

  The differential is the part of the difference that's
  <with|font-shape|italic|linear> in <math|d A>:

  <\equation*>
    d f<around*|(|A;d A|)>=A <around*|(|d A|)>+<around*|(|d A|)> A
  </equation*>

  This can be abbreviated as

  <\equation*>
    d f=A <around*|(|d A|)>+<around*|(|d A|)> A.
  </equation*>

  We can represent this result as matrix multiplication:

  <\eqnarray*>
    <tformat|<table|<row|<cell|d f>|<cell|=>|<cell|A <around*|(|d
    A|)>+<around*|(|d A|)> A>>|<row|<cell|>|<cell|=>|<cell|A <around*|(|d
    A|)>I+I<around*|(|d A|)> A>>|<row|<cell|vec<around*|(|d
    f|)>>|<cell|=>|<cell|vec<around*|(|A <around*|(|d A|)>I+I<around*|(|d
    A|)> A|)>>>|<row|<cell|>|<cell|=>|<cell|vec<around*|(|A <around*|(|d
    A|)>I|)>+vec<around*|(|I<around*|(|d A|)>
    A|)>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|I\<otimes\>A|)>
    vec<around*|(|d A|)>+<around*|(|A<rsup|T>\<otimes\>I|)>vec<around*|(|d
    A|)>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|I\<otimes\>A+A<rsup|T>\<otimes\>I|)>
    vec<around*|(|d A|)>>>>>
  </eqnarray*>

  <section|Example>

  Let <math|f> be the following function that maps from
  <math|\<bbb-R\><rsup|n\<times\>n>> to <math|\<bbb-R\><rsup|n\<times\>n>>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|A|)>>|<cell|=>|<cell|A<rsup|3>>>>>
  </eqnarray*>

  Deriving the difference:

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|f<around*|(|A+d
    A|)>-f<around*|(|A|)>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|A+d
    A|)><rsup|3>-A<rsup|3>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|A+d
    A|)><rsup|2><around*|(|A+d A|)>-A<rsup|3>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|A+d
    A|)><rsup|2>A+<around*|(|A+d A|)><rsup|2> d
    A-A<rsup|3>>>|<row|<cell|>|<cell|=>|<cell|<around*|[|A<rsup|2>+A
    <around*|(|d A|)>+<around*|(|d A|)>A+<around*|(|d
    A|)><rsup|2>|]>A+<around*|[|A<rsup|2>+A <around*|(|d A|)>+<around*|(|d
    A|)> A+<around*|(|d A|)><rsup|2>|]>d A-A<rsup|3>>>|<row|<cell|>|<cell|=>|<cell|A<rsup|3>+A<around*|(|d
    A|)>A+<rsup|><around*|(|d A|)>A<rsup|2>+<around*|(|d
    A|)><rsup|2>A+A<rsup|2><around*|(|d A|)>+A<around*|(|d
    A|)><rsup|2>+<around*|(|d A|)>A<around*|(|d A|)>+<around*|(|d
    A|)><rsup|3>-A<rsup|3>>>|<row|<cell|>|<cell|=>|<cell|A<around*|(|d
    A|)>A+<rsup|><around*|(|d A|)>A<rsup|2>+<around*|(|d
    A|)><rsup|2>A+A<rsup|2><around*|(|d A|)>+A<around*|(|d
    A|)><rsup|2>+<around*|(|d A|)>A<around*|(|d A|)>+<around*|(|d
    A|)><rsup|3>>>>>
  </eqnarray*>

  The differential is the part of the difference that's
  <with|font-shape|italic|linear> in <math|d A>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|d f<around*|(|A;d
    A|)>>|<cell|=>|<cell|A<around*|(|d A|)>A+<around*|(|d
    A|)>A<rsup|2>+A<rsup|2><around*|(|d A|)>>>>>
  </eqnarray*>

  This can be abbreviated as

  <\equation*>
    d f=A<around*|(|d A|)>A+<around*|(|d A|)>A<rsup|2>+A<rsup|2><around*|(|d
    A|)>
  </equation*>

  We can represent this result as matrix multiplication:

  <\eqnarray*>
    <tformat|<table|<row|<cell|d f>|<cell|=>|<cell|A<around*|(|d
    A|)>A+<around*|(|d A|)>A<rsup|2>+A<rsup|2><around*|(|d
    A|)>>>|<row|<cell|>|<cell|=>|<cell|A<around*|(|d A|)>A+I<around*|(|d
    A|)>A<rsup|2>+A<rsup|2><around*|(|d A|)>I>>|<row|<cell|vec<around*|(|d
    f|)>>|<cell|=>|<cell|<around*|(|A<rsup|T>\<otimes\>A+<around*|(|A<rsup|2>|)><rsup|T>\<otimes\>I+I
    \<otimes\>A<rsup|2>|)> vec<around*|(|d A|)>>>>>
  </eqnarray*>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
    <associate|auto-3|<tuple|3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Differential>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Example>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Example>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>