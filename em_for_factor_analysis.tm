<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|EM Algorithm for Factor Analysis>|<\doc-date>
    Feb 24, 2024
  </doc-date>>

  <section|Generative model><label|sec:model>

  Let <math|<with|font-series|bold|z><rsub|i>\<in\>\<bbb-R\><rsup|L>> be the
  latent variable and <math|<with|font-series|bold|x><rsub|i>\<in\>\<bbb-R\><rsup|D>>
  be the observed variable. The factor analysis model is defined as follows:

  <\eqnarray*>
    <tformat|<table|<row|<cell|p<around*|(|<with|font-series|bold|z><rsub|i>|)>>|<cell|=>|<cell|<with|font|cal|N><around*|(|<with|font-series|bold|z><rsub|i>\<mid\><with|font-series|bold|0>,<with|font-series|bold|I><rsub|L>|)>>>|<row|<cell|p<around*|(|<with|font-series|bold|x><rsub|i>\<mid\><with|font-series|bold|z><rsub|i>,<with|font-series|bold|\<theta\>>|)>>|<cell|=>|<cell|<with|font|cal|N><around*|(|<with|font-series|bold|x><rsub|i>\<mid\><with|font-series|bold|W
    z><rsub|i>+<with|font-series|bold|\<mu\>>,<with|font-series|bold|\<Psi\>>|)>>>>>
  </eqnarray*>

  where <math|<with|font-series|bold|W>\<in\>\<bbb-R\><rsup|D\<times\>L>>,
  <math|<with|font-series|bold|\<mu\>>\<in\>\<bbb-R\><rsup|D>> and
  <math|<with|font-series|bold|\<Psi\>>> (diagonal covariance) are the model
  parameters.

  <section|EM algorithm overview>

  As discussed in Section <reference|sec:model>, the model parameters to be
  estimated are <math|<with|font-series|bold|W>,<with|font-series|bold|\<mu\>>,<with|font-series|bold|\<Psi\>>>.
  However, given a dataset, we can simply subtract from each dimension its
  own empirical mean so that <math|<with|font-series|bold|\<mu\>>=0>. Because
  of this, we only need to estimate <math|<with|font-series|bold|W>> and
  <math|<with|font-series|bold|\<Psi\>>>:

  <\equation*>
    <with|font-series|bold|W><rsub|t+1>,<with|font-series|bold|\<Psi\>><rsub|t+1>\<leftarrow\><wide*|arg
    max<rsub|<with|font-series|bold|W>,<with|font-series|bold|\<Psi\>>>|\<wide-underbrace\>><rsub|<text|M-step>><wide*|<big|sum><rsub|i=1><rsup|n>\<bbb-E\><rsub|<with|font-series|bold|z><rsub|i>\<sim\>p<around*|(|<with|font-series|bold|z><rsub|i>\<mid\><with|font-series|bold|x><rsub|i>,<with|font-series|bold|W><rsub|t>,<with|font-series|bold|\<Psi\>><rsub|t>|)>><around*|[|log
    p<around*|(|<with|font-series|bold|x><rsub|i>,<with|font-series|bold|z><rsub|i>\<mid\><with|font-series|bold|W>,<with|font-series|bold|\<Psi\>>|)>|]>|\<wide-underbrace\>><rsub|<text|E-step>>
  </equation*>

  <section|Deriving EM for FA>

  Deriving EM for a model generally follows two steps:

  <\itemize>
    <item>Simplifying the expression for E-step

    <item>Setting the derivative of that expression w.r.t
    <math|<with|font-series|bold|W>> and <math|<with|font-series|bold|\<Psi\>>>
    to zero and solve for <math|<with|font-series|bold|W>> and
    <math|<with|font-series|bold|\<Psi\>>>
  </itemize>

  \;

  \;

  <section*|References>

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
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
    <associate|auto-3|<tuple|3|?>>
    <associate|auto-4|<tuple|<with|mode|<quote|math>|\<bullet\>>|?>>
    <associate|auto-5|<tuple|3.1|?>>
    <associate|sec:model|<tuple|1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Generative
      model> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>EM
      algorithm overview> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Deriving
      EM for FA> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|References>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>