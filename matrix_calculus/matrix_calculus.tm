<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Matrix Calculus>>

  <section|Lecture 1 Part 2: Derivatives as Linear Operators>

  <subsection|Deriative in single-variable calculus>

  In single-variable calculus, we defined the derivative
  <math|f<rprime|'><around*|(|x|)>> as the slope of the tangent to <math|f>
  at <math|<around*|(|x,f<around*|(|x|)>|)>>. Here, we emphasize that the
  tangent to <math|f> at <math|x> can also be interpreted as a
  <with|font-shape|italic|linear approximation> of <math|f> at <math|x>. In
  other words, we can estimate <math|f<around*|(|x+\<delta\>x|)>> using the
  following formula:

  <\eqnarray*>
    <tformat|<table|<row|<cell|f<around*|(|x+\<delta\>x|)>>|<cell|\<approx\>>|<cell|f<around*|(|x|)>+f<rprime|'><around*|(|x|)>
    \<delta\>x.>>>>
  </eqnarray*>

  So far so good. Now, if we define <math|\<delta\>f=f<around*|(|x+\<delta\>x|)>-f<around*|(|x|)>>
  as the small finite change in <math|f>, we have

  <\equation*>
    \<delta\>f=f<around*|(|x+\<delta\>x|)>-f<around*|(|x|)>\<approx\>f<rprime|'><around*|(|x|)>
    \<delta\>x.
  </equation*>

  As <math|\<delta\>x\<rightarrow\>\<infty\>>, we have

  <\equation*>
    d f=f<around*|(|x+d x|)>-f<around*|(|x|)>=f<rprime|'><around*|(|x|)> d x,
  </equation*>

  where <math|\<delta\>x> and <math|\<delta\>f> have changed to <math|d x>
  and <math|d f> respectively, and <math|\<approx\>> has changed to an
  equality.\ 

  <\equation*>
    <frac|\<partial\>x<rsup|Td>|>
  </equation*>

  \;

  \;

  \;

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
    <associate|auto-1|<tuple|1|?|../../../.TeXmacs/texts/scratch/no_name_28.tm>>
    <associate|auto-2|<tuple|1.1|?|../../../.TeXmacs/texts/scratch/no_name_28.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Lecture
      1 Part 2: Derivatives as Linear Operators>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Deriative in single-variable
      calculus <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>
    </associate>
  </collection>
</auxiliary>