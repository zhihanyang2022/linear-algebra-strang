<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Lecture 1 Part 2: Derivatives as Linear
  Operator>|<\doc-subtitle>
    MIT 18.S096 Matrix Calculus For Machine Learning and Beyond
  </doc-subtitle>|<doc-date|March 17, 2024>>

  <\table-of-contents|toc>
    <vspace*|1fn><with|font-series|bold|math-font-series|bold|1<space|2spc>Differential>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-1><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|2<space|2spc>Example>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-2><vspace|0.5fn>
  </table-of-contents>

  <section|Differential>

  In the scalar-to-scalar case:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|x+d
    x|)>>|<cell|=>|<cell|f<around*|(|x|)>+f<rprime|'><around*|(|x|)> d
    x+<around*|(|<text|higher order terms>|)>>>>>
  </eqnarray*>

  When <math|<with|font-series|bold|x>\<in\>\<bbb-R\><rsup|n>> and
  <math|f:\<bbb-R\><rsup|n>\<rightarrow\>\<bbb-R\>>:

  <\equation*>
    f<around*|(|<with|font-series|bold|x>+d<with|font-series|bold|x>|)>=f<around*|(|<with|font-series|bold|x>|)>+f<rprime|'><around*|(|<with|font-series|bold|x>|)>d
    <with|font-series|bold|x>+<around*|(|<text|higher order terms>|)>
  </equation*>

  We see that the derivative <math|f<rprime|'><around*|(|<with|font-series|bold|x>|)>>
  must be a row vector in order for <math|f<rprime|'><around*|(|<with|font-series|bold|x>|)>d
  <with|font-series|bold|x>> to be a scalar. We define the gradient
  <math|\<nabla\>f> as the column vector <math|f<rprime|'><around*|(|<with|font-series|bold|x>|)><rsup|T>>.\ 

  <section|Example>

  Let <math|f:\<bbb-R\><rsup|n>\<rightarrow\>\<bbb-R\>> be the function
  below:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|<with|font-series|bold|x>|)>>|<cell|=>|<cell|<with|font-series|bold|x><rsup|T>A<with|font-series|bold|x>.>>>>
  </eqnarray*>

  Calculating the difference, the differential and the gradient:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|<with|font-series|bold|x>+d<with|font-series|bold|x>|)>-f<around*|(|<with|font-series|bold|x>|)>>|<cell|=>|<cell|<around*|(|<with|font-series|bold|x>+d<with|font-series|bold|x>|)><rsup|T>A<around*|(|<with|font-series|bold|x>+d<with|font-series|bold|x>|)>-<with|font-series|bold|x><rsup|T>A<with|font-series|bold|x>>>|<row|<cell|>|<cell|=>|<cell|<with|font-series|bold|x><rsup|T>A<with|font-series|bold|x>+<with|font-series|bold|x><rsup|T>A<around*|(|d<with|font-series|bold|x>|)>+<around*|(|d<with|font-series|bold|x>|)><rsup|T>A<with|font-series|bold|x>+<around*|(|d<with|font-series|bold|x>|)><rsup|T>A<around*|(|d<with|font-series|bold|x>|)>-<with|font-series|bold|x><rsup|T>A<with|font-series|bold|x>>>|<row|<cell|>|<cell|=>|<cell|<with|font-series|bold|x><rsup|T>A<around*|(|d<with|font-series|bold|x>|)>+<around*|(|d<with|font-series|bold|x>|)><rsup|T>A<with|font-series|bold|x>+<around*|(|d<with|font-series|bold|x>|)><rsup|T>A<around*|(|d<with|font-series|bold|x>|)>>>|<row|<cell|d
    f>|<cell|=>|<cell|<with|font-series|bold|x><rsup|T>A<around*|(|d<with|font-series|bold|x>|)>+<around*|(|d<with|font-series|bold|x>|)><rsup|T>A<with|font-series|bold|x>>>|<row|<cell|>|<cell|=>|<cell|<with|font-series|bold|x><rsup|T>A<around*|(|d<with|font-series|bold|x>|)>+<with|font-series|bold|x><rsup|T>A<rsup|T><around*|(|d<with|font-series|bold|x>|)>>>|<row|<cell|>|<cell|=>|<cell|<with|font-series|bold|x><rsup|T><around*|(|A+A<rsup|T>|)><around*|(|d<with|font-series|bold|x>|)>>>|<row|<cell|\<nabla\>f>|<cell|=>|<cell|<around*|(|A+A<rsup|T>|)><with|font-series|bold|x>>>>>
  </eqnarray*>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|papyrus>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|?>>
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
    </associate>
  </collection>
</auxiliary>