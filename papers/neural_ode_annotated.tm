<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Implementation Notes on Neural ODE>>

  <section|Background>

  A dynamical system can be described by its initial state and how that state
  changes over time. We are interested in the system's state after some time.\ 

  <section|JAX vjp>

  jax.vjp(fun, *primals, hax_aux=False, reduce_axes=())

  <\python-code>
    def aug_dynamics(augmented_state, t, theta):

    \ \ \ \ z, a, a_t, a_theta = augmented_state

    \ \ \ \ y_dot, vjpfun = jax.vjp(func, y, -t, *args)

    \ \ \ \ return (-y_dot, *vjpfun(y_bar))
  </python-code>

  <section|Reverse-mode automatic differentiation of ODE solutions>

  <subsection|Forward pass>

  Initial state at <math|t<rsub|0>>:

  <\equation*>
    <with|font-series|bold|z><around*|(|t<rsub|0>|)>
  </equation*>

  ODE:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|d <with|font-series|bold|z><around*|(|t|)>|d
    t>>|<cell|=>|<cell|f<rsub|><around*|(|<with|font-series|bold|z><around*|(|t|)>,t,<with|font-series|bold|\<theta\>>|)>>>>>
  </eqnarray*>

  Final state at <math|t<rsub|1>>:

  <\equation*>
    ODESolve<around*|(|<with|font-series|bold|z><around*|(|t<rsub|0>|)>,f,t<rsub|0>,t<rsub|1>,<with|font-series|bold|\<theta\>>|)>
  </equation*>

  <subsection|Backward pass>

  <subsubsection|Dynamics of adjoint state>

  Define the <with|font-shape|italic|adjoint> state:

  <\equation*>
    <with|font-series|bold|a><around*|(|t|)>=<frac|d L|d
    <with|font-series|bold|z><around*|(|t|)>>
  </equation*>

  Final state at <math|t<rsub|1>>:

  <\equation*>
    <with|font-series|bold|a><around*|(|t<rsub|1>|)>=<frac|d L|d
    <with|font-series|bold|z><around*|(|t<rsub|1>|)>>
  </equation*>

  ODE (proved in the paper):

  <\equation*>
    <frac|d <with|font-series|bold|a><around*|(|t|)>|d
    t>=-<with|font-series|bold|a><around*|(|t|)><rsup|T>
    <frac|\<partial\>f<around*|(|<with|font-series|bold|z><around*|(|t|)>,t,\<theta\>|)>|\<partial\>
    <with|font-series|bold|z><around*|(|t|)>>
  </equation*>

  Initial state at <math|t<rsub|0>> (exact):

  <\equation*>
    <with|font-series|bold|a><around*|(|t<rsub|0>|)>=<with|font-series|bold|a><around*|(|t<rsub|1>|)>+<big|int><rsub|t<rsub|1>><rsup|t<rsub|0>><frac|d
    <with|font-series|bold|a><around*|(|t|)>|d t> d
    t=<with|font-series|bold|a><around*|(|t<rsub|1>|)>-<big|int><rsub|t<rsub|1>><rsup|t<rsub|0>><with|font-series|bold|a><around*|(|t|)><rsup|T>
    <frac|\<partial\>f<around*|(|<with|font-series|bold|z><around*|(|t|)>,t,\<theta\>|)>|\<partial\>
    <with|font-series|bold|z><around*|(|t|)>> d t
  </equation*>

  Initial state at <math|t<rsub|0>> (approximate):

  <\equation*>
    <with|font-series|bold|a><around*|(|t<rsub|0>|)>\<approx\>ODESolve<around*|(|<with|font-series|bold|a><around*|(|t<rsub|1>|)>,-<with|font-series|bold|a><around*|(|t|)><rsup|T>
    <frac|\<partial\>f<around*|(|<with|font-series|bold|z><around*|(|t|)>,t,\<theta\>|)>|\<partial\>
    <with|font-series|bold|z><around*|(|t|)>>,t<rsub|1>,t<rsub|0>,\<theta\>|)>
  </equation*>

  <subsubsection|Gradient wrt. <math|\<theta\>> and <math|t>>

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
    <associate|auto-1|<tuple|1|?|../../../../.TeXmacs/texts/scratch/no_name_31.tm>>
    <associate|auto-2|<tuple|2|?|../../../../.TeXmacs/texts/scratch/no_name_31.tm>>
    <associate|auto-3|<tuple|3|?|../../../../.TeXmacs/texts/scratch/no_name_31.tm>>
    <associate|auto-4|<tuple|3.1|?|../../../../.TeXmacs/texts/scratch/no_name_31.tm>>
    <associate|auto-5|<tuple|3.2|?|../../../../.TeXmacs/texts/scratch/no_name_31.tm>>
    <associate|auto-6|<tuple|3.2.1|?|../../../../.TeXmacs/texts/scratch/no_name_31.tm>>
    <associate|auto-7|<tuple|3.2.2|?|../../../../.TeXmacs/texts/scratch/no_name_31.tm>>
    <associate|auto-8|<tuple|2.2.3|?|../../../../.TeXmacs/texts/scratch/no_name_31.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Background>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Reverse-mode
      automatic differentiation of ODE solutions>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Forward pass
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Backward pass
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|2tab>|2.2.1<space|2spc>Dynamics of adjoint state
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|2tab>|2.2.2<space|2spc>Dynamics of parameters
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|2tab>|2.2.3<space|2spc>Dynamics of time
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>
    </associate>
  </collection>
</auxiliary>