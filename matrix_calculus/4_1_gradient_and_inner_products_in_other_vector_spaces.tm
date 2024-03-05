<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Lecture 4 Part 1: Gradients and Inner Products in
  Other Vector Spaces>|<doc-subtitle|MIT 18.S096 Matrix Calculus For Machine
  Learning And Beyond>|<doc-date|March 5, 2024>>

  <\table-of-contents|toc>
    <vspace*|1fn><with|font-series|bold|math-font-series|bold|Riesz
    representation theorem> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-1><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|Example:
    gradient of <with|mode|math|<around*|\<\|\|\>|A|\<\|\|\>><rsub|F>> with
    respect to <with|mode|math|A>> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-2><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|Example:
    gradient of <with|mode|math|x<rsup|T>A y> with respect to
    <with|mode|math|A>> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-3><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|Example:
    gradient of <with|mode|math|sum<around*|(|A|)>> with respect to
    <with|mode|math|A>> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-4><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|Lingering
    questions> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-5><vspace|0.5fn>
  </table-of-contents>

  <section*|Riesz representation theorem>

  A <with|font-shape|italic|Hilbert space> is a continuous vector space with
  an inner (dot/scalar) product defined. For <math|\<bbb-R\><rsup|n>> (i.e.,
  column vectors), we usually define the inner product as
  <math|<with|font-series|bold|x>\<cdot\><with|font-series|bold|y>=<with|font-series|bold|x><rsup|T><with|font-series|bold|y>>.
  For <math|\<bbb-R\><rsup|n\<times\>m>> (i.e., matrices), we usually define
  the inner product as <math|sum<around*|(|<with|font-series|bold|A>\<odot\><with|font-series|bold|B>|)>=<around*|(|vec<around*|(|A|)>|)><rsup|T><around*|(|vec
  B|)>=tr<around*|(|A<rsup|T>B|)>>. Three properties of a valid inner
  product:

  <\enumerate>
    <item>Symmetric: <math|x\<cdot\>y=y\<cdot\>x>

    <item>Linear: <math|x\<cdot\><around*|(|\<alpha\>y+\<beta\>z|)>=\<alpha\><around*|(|x\<cdot\>y|)>+\<beta\><around*|(|x\<cdot\>z|)>>

    <item>Non-negative: <math|x\<cdot\>x=<around*|\<\|\|\>|x|\<\|\|\>><rsup|2>\<geq\>0>,
    <math|=0> iff <math|x=0>
  </enumerate>

  <with|font-shape|italic|Setup.> Let <math|f<around*|(|x|)>> be a function
  that maps from a Hilbert space to <math|\<bbb-R\>>. We know that the
  derivative is the linear operator (\Plinear form\Q) that takes a <math|d x>
  (a infinitesimal change in the input) to <math|d f> (a infinitesimal change
  in the output):

  <\equation*>
    d f=f<rprime|'><around*|(|x|)><around*|[|d x|]>.
  </equation*>

  <with|font-shape|italic|Riesz representation theorem> tells us that if we
  have a linear function that's \Pvector in number out\Q, then it can be
  represented as a dot product with its input. So
  <math|f<rprime|'><around*|(|x|)><around*|[|d x|]>> can be represented as
  the dot product between some vector and <math|d x>, and we call this vector
  the <with|font-shape|italic|gradient>:

  <\equation*>
    d f=f<rprime|'><around*|(|x|)><around*|[|d
    x|]>=<around*|(|\<nabla\>f|)>\<cdot\><around*|(|d x|)>.
  </equation*>

  An observation is that the gradient would always have the same \Pshape\Q as
  <math|x>.

  <with|font-shape|italic|General strategy from deriving the gradient.> Start
  with <math|d f>. Gradually massage it into a dot product between something
  and <math|d x>. That \Psomething\Q would be the gradient.\ 

  Below we show some examples of this procedure from matrix-scalar functions.
  Note that the stuff on the LHS is equivalent to the stuff on the RHS, which
  is written in 18.06 style:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<tabular*|<tformat|<cwith|1|-1|1|-1|cell-halign|l>|<table|<row|<cell|d
    f>|<cell|=>|<cell|<around*|(|\<nabla\>f|)>\<cdot\>dA>>|<row|<cell|>|<cell|=>|<cell|tr<around*|(|<around*|(|\<nabla\>f|)><rsup|T>d
    A|)>>>>>>>|<cell|<below|\<Leftrightarrow\>|equivalent>>|<cell|\<nabla\>f=<matrix|<tformat|<table|<row|<cell|<frac|\<partial\>f|\<partial\>A<rsub|11>>>|<cell|<frac|\<partial\>f|\<partial\>A<rsub|12>>>|<cell|\<cdots\>>>|<row|<cell|<frac|\<partial\>f|\<partial\>A<rsub|21>>>|<cell|<frac|\<partial\>f|\<partial\>A<rsub|22>>>|<cell|\<cdots\>>>|<row|<cell|\<vdots\>>|<cell|\<vdots\>>|<cell|\<ddots\>>>>>>>>>>
  </eqnarray*>

  <section*|Example: gradient of <math|<around*|\<\|\|\>|A|\<\|\|\>><rsub|F>>
  with respect to <math|A>>

  Function:

  <\equation*>
    f<around*|(|A<rsub|m\<times\>n>|)>=<around*|\<\|\|\>|A|\<\|\|\>><rsub|F>=<sqrt|tr<around*|(|A<rsup|T>A|)>>
  </equation*>

  Deriving the gradient:

  <\eqnarray*>
    <tformat|<table|<row|<cell|d f>|<cell|=>|<cell|<frac|1|2<sqrt|tr<around*|(|A<rsup|T>A|)>>>
    d<around*|(|tr<around*|(|A<rsup|T>A|)>|)><space|1em><around*|(|<text|scalar
    chain rule>|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|2<sqrt|tr<around*|(|A<rsup|T>A|)>>>
    tr<around*|[|d<around*|(|A<rsup|T>A|)>|]><space|1em><around*|(|<text|trace
    is linear>|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|2<sqrt|tr<around*|(|A<rsup|T>A|)>>>tr<around*|[|<around*|(|d
    A|)><rsup|T>A+A<rsup|T>d A|]><space|1em><around*|(|<text|matrix chain
    rule>|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|2<sqrt|tr<around*|(|A<rsup|T>A|)>>><around*|<left|(|2>|tr<around*|[|<around*|(|d
    A|)><rsup|T>A|]>+tr<around*|(|A<rsup|T>d
    A|)>|<right|)|2>>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|2<sqrt|tr<around*|(|A<rsup|T>A|)>>><around*|<left|(|2>|tr<around*|[|A<rsup|T><around*|(|d
    A|)>|]>+tr<around*|(|A<rsup|T>d A|)>|<right|)|2>><space|1em><around*|(|tr<around*|(|A|)>=tr<around*|(|A<rsup|T>|)>|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|<sqrt|tr<around*|(|A<rsup|T>A|)>>>
    tr<around*|[|A<rsup|T><around*|(|d A|)>|]><space|1em>>>|<row|<cell|>|<cell|=>|<cell|<frac|1|<sqrt|tr<around*|(|A<rsup|T>A|)>>>
    A\<cdot\>d A<space|1em><around*|(|<text|definition of the matrix dot
    product>|)>>>|<row|<cell|>|<cell|=>|<cell|<wide*|<frac|A|<sqrt|tr<around*|(|A<rsup|T>A|)>>>|\<wide-underbrace\>><rsub|gradient>\<cdot\>d
    A>>>>
  </eqnarray*>

  Note that the gradient is simply <math|A> divided by
  <math|<around*|\<\|\|\>|A|\<\|\|\>><rsub|F>>.

  <section*|Example: gradient of <math|x<rsup|T>A y> with respect to
  <math|A>>

  Function:

  <\equation*>
    f<around*|(|A<rsub|m\<times\>n>|)>=x<rsup|T>A y
  </equation*>

  for some constant <math|x\<in\>\<bbb-R\><rsup|m>> and
  <math|y\<in\>\<bbb-R\><rsup|n>>.

  Deriving the gradient:

  <\eqnarray*>
    <tformat|<table|<row|<cell|d f>|<cell|=>|<cell|x<rsup|T>d A
    y<space|1em>*<around*|(|<text|matrix product
    rule>|)>>>|<row|<cell|>|<cell|=>|<cell|tr<around*|(|x<rsup|T>d A
    y|)>>>|<row|<cell|>|<cell|=>|<cell|tr<around*|(|y x<rsup|T>d
    A|)>>>|<row|<cell|>|<cell|=>|<cell|<wide*|<around*|(|x
    y<rsup|T>|)>|\<wide-underbrace\>><rsub|gradient>\<cdot\>d A>>>>
  </eqnarray*>

  <section*|Example: gradient of <math|sum<around*|(|A|)>> with respect to
  <math|A>>

  Function:

  <\equation*>
    f<around*|(|A<rsub|m\<times\>n>|)>=sum<around*|(|A|)>=<with|font-series|bold|1><rsup|T>A<with|font-series|bold|1>=sum<around*|(|matrix<around*|(|1|)>\<odot\>A|)>
  </equation*>

  Deriving the gradient (two ways are pretty much equivalent):

  First way (using <math|<with|font-series|bold|1><rsup|T>A<with|font-series|bold|1>>):

  <\eqnarray*>
    <tformat|<table|<row|<cell|d f>|<cell|=>|<cell|d<around*|(|<with|font-series|bold|1><rsup|T>A<with|font-series|bold|1>|)>>>|<row|<cell|>|<cell|=>|<cell|<with|font-series|bold|1><rsup|T>d
    A<with|font-series|bold|1>>>|<row|<cell|>|<cell|=>|<cell|tr<around*|(|<with|font-series|bold|1><rsup|T>d
    A<with|font-series|bold|1>|)>>>|<row|<cell|>|<cell|=>|<cell|tr<around*|(|<with|font-series|bold|1><with|font-series|bold|1><rsup|T>d
    A|)>>>|<row|<cell|>|<cell|=>|<cell|<wide*|<around*|(|<with|font-series|bold|1><with|font-series|bold|1><rsup|T>|)>|\<wide-underbrace\>><rsub|gradient>\<cdot\>d
    A>>>>
  </eqnarray*>

  Second way (using <math|sum<around*|(|matrix<around*|(|1|)>\<odot\>A|)>>):

  <\eqnarray*>
    <tformat|<table|<row|<cell|d f>|<cell|=>|<cell|sum<around*|(|matrix<around*|(|1|)>\<odot\>d
    A|)><space|1em><around*|(|<text|both sum and hadamard product are
    linear>|)>>>|<row|<cell|>|<cell|=>|<cell|<wide*|matrix<around*|(|1|)>|\<wide-underbrace\>><rsub|gradient>\<cdot\>d
    A>>>>
  </eqnarray*>

  <section*|Lingering questions>

  <\itemize>
    <item>What would be the interpretation of the gradient if I define a
    weird but valid inner product?
  </itemize>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|?|1>>
    <associate|auto-2|<tuple|3|2>>
    <associate|auto-3|<tuple|3|2>>
    <associate|auto-4|<tuple|3|2>>
    <associate|auto-5|<tuple|3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Riesz
      representation theorem> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Example:
      gradient of <with|mode|<quote|math>|<around*|\<\|\|\>|A|\<\|\|\>><rsub|F>>
      with respect to <with|mode|<quote|math>|A>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Example:
      gradient of <with|mode|<quote|math>|x<rsup|T>A y> with respect to
      <with|mode|<quote|math>|A>> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Example:
      gradient of <with|mode|<quote|math>|sum<around*|(|A|)>> with respect to
      <with|mode|<quote|math>|A>> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Lingering
      questions> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>