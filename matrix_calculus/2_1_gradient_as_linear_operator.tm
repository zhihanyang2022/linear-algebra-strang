<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Lecture 2 Part 1: Derivatives as Linear
  Operator>|<\doc-subtitle>
    MIT 18.S096 Matrix Calculus For Machine Learning and Beyond
  </doc-subtitle>|<doc-date|March 17, 2024>>

  <section|Differential>

  In the 1D case:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|x+d
    x|)>>|<cell|=>|<cell|f<around*|(|x|)>+f<rprime|'><around*|(|x|)> d
    x+<around*|(|<text|higher order terms>|)>>>>>
  </eqnarray*>

  When <math|v\<in\>\<bbb-R\><rsup|n>> and
  <math|f:\<bbb-R\><rsup|n>\<rightarrow\>\<bbb-R\>>:

  <\equation*>
    f<around*|(|v+d v|)>=f<around*|(|v|)>+f<rprime|'><around*|(|v|)>+<around*|(|<text|higher
    order term>s|)>
  </equation*>

  \;

  <\equation*>
    <frac|\<partial\>l|\<partial\>A<rsub|i
    j>>=<big|sum><rsub|k,l><frac|\<partial\>l|\<partial\>C<rsub|k
    l>>\<cdot\><frac|\<partial\>C<rsub|k l>|\<partial\>A<rsub|i j>>
  </equation*>
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
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Differential>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>