<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Forward-mode and Reverse-mode Automatic
  Differentiaton>>

  Without loss of generality, consider the following computation graph or
  DAG:

  <\big-figure|<image|computation_graph.pdf|0.75par|||>>
    \;
  </big-figure>

  A central goal<\footnote>
    There are other goals such as obtaining a Jacobian-vector product. We
    will discuss these other goals later.
  </footnote> of automatic differentiation is to obtain the Jacobian

  <\equation*>
    J\<triangleq\><matrix|<tformat|<table|<row|<cell|<frac|\<partial\>y<rsub|1>|\<partial\>x<rsub|1>>>|<cell|<frac|\<partial\>y<rsub|1>|\<partial\>x<rsub|2>>>|<cell|<frac|\<partial\>y<rsub|1>|\<partial\>x<rsub|3>>>>|<row|<frac|\<partial\>y<rsub|2>|\<partial\>x<rsub|1>>|<cell|<frac|\<partial\>y<rsub|2>|\<partial\>x<rsub|2>>>|<cell|<frac|\<partial\>y<rsub|2>|\<partial\>x<rsub|3>>>>|<row|<cell|<frac|\<partial\>y<rsub|3>|\<partial\>x<rsub|1>>>|<cell|<frac|\<partial\>y<rsub|3>|\<partial\>x<rsub|2>>>|<cell|<frac|\<partial\>y<rsub|3>|\<partial\>x<rsub|3>>>>>>>.
  </equation*>

  There are two fundamental ways of viewing a computation graph:

  <tabular*|<tformat|<twith|table-width|1par>|<twith|table-hmode|exact>|<table|<row|<cell|<image|forward_mode_view.pdf|0.45par|||>>|<cell|<image|reverse_mode_view.pdf|0.45par|||>>>|<row|<cell|View
  1: as many-parents-one-child structures>|<cell|View 2: as
  one-parent-many-children structures>>>>>

  Each view relates to the goal of automatic differentiation (obtaining the
  Jacobian matrix for now) in an interesting way.\ 

  Consider the first view, the many-parents-one-child view. Let
  <math|x<rsub|i>> be the child. Then by chain rule, we see that
  <math|\<nabla\>x<rsub|i>> only depends on 2 things: (1) the partial
  derivatives <math|\<partial\>x<rsub|i>/\<partial\>x<rsub|j>>'s and (2) the
  <math|\<nabla\>x<rsub|j>>'s. Therefore, if we start from the left-side of
  the computational graph and sweep right, eventually we would get
  <math|\<nabla\>y<rsub|1>>, <math|\<nabla\>y<rsub|2>> and
  <math|\<nabla\>y<rsub|3>> stored in the rightmost nodes.

  <\equation*>
    <matrix|<tformat|<table|<row|<cell|<frac|\<partial\>x<rsub|i>|\<partial\>x<rsub|1>>>>|<row|<cell|<frac|\<partial\>x<rsub|i>|\<partial\>x<rsub|2>>>>|<row|<cell|<frac|\<partial\>x<rsub|i>|\<partial\>x<rsub|3>>>>>>>=<big|sum><rsub|j\<in\>parents<around*|(|i|)>><frac|\<partial\>x<rsub|i>|\<partial\>x<rsub|j>>\<cdot\>
    <matrix|<tformat|<table|<row|<cell|<frac|\<partial\>x<rsub|j>|\<partial\>x<rsub|1>>>>|<row|<cell|<frac|\<partial\>x<rsub|j>|\<partial\>x<rsub|2>>>>|<row|<cell|<frac|\<partial\>x<rsub|j>|\<partial\>x<rsub|3>>>>>>>.
  </equation*>

  Now consider the second view. Let <math|x<rsub|i>> be the parent. Then, by
  chain rule, we have

  <\equation*>
    <matrix|<tformat|<table|<row|<cell|<frac|\<partial\>y<rsub|1>|\<partial\>x<rsub|i>>>>|<row|<cell|<frac|\<partial\>y<rsub|2>|\<partial\>x<rsub|i>>>>|<row|<cell|<frac|\<partial\>y<rsub|3>|\<partial\>x<rsub|i>>>>>>>=<big|sum><rsub|j\<in\>children<around*|(|i|)>><matrix|<tformat|<table|<row|<cell|<frac|\<partial\>y<rsub|1>|\<partial\>x<rsub|j>>>>|<row|<cell|<frac|\<partial\>y<rsub|2>|\<partial\>x<rsub|j>>>>|<row|<cell|<frac|\<partial\>y<rsub|3>|\<partial\>x<rsub|j>>>>>>>\<cdot\><frac|\<partial\>x<rsub|j>|\<partial\>x<rsub|i>>
  </equation*>

  Each node stores:

  <\itemize>
    <item>Partial derivatives to its children

    <item>Reference to its children (to get
    <math|<matrix|<tformat|<table|<row|<cell|<frac|\<partial\>y<rsub|1>|\<partial\>x<rsub|j>>>>|<row|<cell|<frac|\<partial\>y<rsub|2>|\<partial\>x<rsub|j>>>>|<row|<cell|<frac|\<partial\>y<rsub|3>|\<partial\>x<rsub|j>>>>>>>>)
  </itemize>

  \;

  The core of understanding this topic, would be how these primal and
  cotangent information work for scalars, vectors, matrices and other
  objects!

  Understanding the matrix-matrix product would be crucially important

  Might need to review differential geometry to continue this

  Note that not knowing the derivation does not prevent me from implementing
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|footnote-1|<tuple|1|1>>
    <associate|footnr-1|<tuple|1|1>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|figure>
      <tuple|normal|<\surround|<hidden-binding|<tuple>|1>|>
        \;
      </surround>|<pageref|auto-1>>
    </associate>
  </collection>
</auxiliary>