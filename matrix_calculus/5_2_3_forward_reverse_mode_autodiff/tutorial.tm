<TeXmacs|2.1.1>

<style|<tuple|generic|framed-program>>

<\body>
  <doc-data|<doc-title|First-Order Automatic Differentiation in
  JAX>|<doc-subtitle|An In-Depth Tutorial>|<doc-author|<author-data|<\author-affiliation>
    <with|font-series|bold|Zhihan Yang>

    \;

    Department of Computer Science

    Cornell University

    Ithaca, NY 14853

    <verbatim|zhihany@cornell.edu>
  </author-affiliation>>>>

  <abstract-data|<\abstract>
    JAX is a high-performance numerical-computing library in Python recently
    developed by Google. With a syntax similar to Numpy, it also features
    just-in-time compilation, automatic differentiation (AD), and hardware
    acceleration via XLA<\footnote>
      According to Google video, this is how JAX got its name
    </footnote>. This tutorial explores two critical aspects of JAX's AD
    system for computing first-order derivatives<\footnote>
      Also called gradients. We do not include hessian computation this in
      tutorial.
    </footnote>.\ 

    Generally, we want use AD to find the derivative of each output scalar
    variable<\footnote>
      These scalar variables may be organized in vectors, matrices or
      tensors.
    </footnote> with respect to each scalar input variable of a large acyclic
    computational graph composed of simpler functions with their own inputs
    and outputs. Most existing tutorials on AD assume that both the
    computation graph and its consisting functions take in one scalar/vector
    and outputs one scalar/vector. While this is fine for pedagogy, it is not
    realistic since, in practice, functions can take in and output a
    list<\footnote>
      Tree is an interesting thing
    </footnote> of tensors. Indeed, the programming abstractions of JAX's AD
    system were developed to compute derivatives of functions of such
    generality, and can hence be cryptic for those who only understand AD for
    scalar-to-scalar and vector-to-vector functions.\ 

    We first explain how deratives of such a computation graph can be
    computed via forward-mode and reverse-mode AD; important concepts such as
    Jaocbians, Jacobian-vector products and vector-Jacobian products are
    discussed in great detail. Cmoputationa graph input and output. Each
    function input and output. We then explain how users can supply custom AD
    rules to differentiate through functions that, e.g., does not include JAX
    code. Finally, we derive from scratch AD rules for common functions.\ 

    JAX is sophisticated codebase and we will not go to the source code.
    Rather, it gives the reader a mental model of how common JAX's functions
    work under the hood and relate to other. Prepare the fully unlease the
    potential JAX for machine learning research.
  </abstract>>

  <\table-of-contents|toc>
    <vspace*|1fn><with|font-series|bold|math-font-series|bold|1<space|2spc>Multivariate
    chain rule> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-1><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|2<space|2spc>The
    goal of automatic differentiation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-2><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|3<space|2spc>Forward-mode
    automatic differentiation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-3><vspace|0.5fn>

    <with|par-left|1tab|3.1<space|2spc>Understanding
    <with|font-family|tt|language|verbatim|jac.jacfwd>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-4>>

    <with|par-left|1tab|3.2<space|2spc>Understanding
    <with|font-family|tt|language|verbatim|jac.jvp>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-5>>

    <with|par-left|1tab|3.3<space|2spc>How JAX uses
    <with|font-family|tt|language|verbatim|jac.jvp> to compute the
    \PJacobian\Q ? <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-6>>

    <with|par-left|1tab|3.4<space|2spc>Defining custom JVP rules /
    pushforward rules via <with|font-family|tt|language|verbatim|jax.custom_jvp>
    and <with|font-family|tt|language|verbatim|f.defjvp>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-7>>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|4<space|2spc>Reverse-mode
    automatic differentiation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-8><vspace|0.5fn>

    <with|par-left|1tab|4.1<space|2spc>Computing the full \PJacobian\Q
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-9>>

    <with|par-left|1tab|4.2<space|2spc>Understanding
    <with|font-family|tt|language|verbatim|jax.vjp>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-10>>

    <with|par-left|1tab|4.3<space|2spc>Understanding
    <with|font-family|tt|language|verbatim|jax.jacrev>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-11>>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|5<space|2spc>Comparing
    forward mode and reverse mode> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-12><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|6<space|2spc>Derivations
    of some JVP / pushforward rules> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-13><vspace|0.5fn>

    <with|par-left|1tab|6.1<space|2spc>Scalar addition
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-14>>

    <with|par-left|1tab|6.2<space|2spc>Scalar multiplication
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-15>>

    <with|par-left|1tab|6.3<space|2spc>Scalar sine
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-16>>

    <with|par-left|1tab|6.4<space|2spc>Broadcasted function
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-17>>

    <with|par-left|1tab|6.5<space|2spc>Matrix-vector product
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-18>>

    <with|par-left|1tab|6.6<space|2spc>Scalar root-finding
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-19>>

    <with|par-left|1tab|6.7<space|2spc>Matrix-matrix product
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-20>>

    <with|par-left|1tab|6.8<space|2spc>L2 loss
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-21>>

    <with|par-left|1tab|6.9<space|2spc>Linear system
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-22>>

    <with|par-left|1tab|6.10<space|2spc>Nonlinear system solve
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-23>>

    <with|par-left|1tab|6.11<space|2spc>Neural ODE
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-24>>

    <with|par-left|1tab|6.12<space|2spc>Softmax
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-25>>
  </table-of-contents>

  <section|Multivariate chain rule>

  The most important theorem for understanding automatic differentiation is
  the multivariate chain rule. In this section, we present three versions of
  this rule with increasing generality. In a college-level multivariate
  calculus class, one would learn the following version of this rule: if four
  scalar variables <math|x,y,z> and <math|t> follow the relationship

  <\equation*>
    t\<mapsto\>x<space|1em>t\<mapsto\>y<space|1em><around*|(|x,y|)>\<mapsto\>z,
  </equation*>

  then the derivative of <math|z> with respect to <math|t> can be calculated
  as

  <\equation>
    <frac|d z|d t>=<frac|\<partial\> z|\<partial\> x> <frac|d x|d
    t>+<frac|\<partial\> z|\<partial\> y> <frac|d y|d
    t>.<label|eq:scalar-chain>
  </equation>

  Its proof can be found in any standard calculus textbook.\ 

  <with|font-series|bold|First generalization.> We can generalize this
  version a bit more. If vectors <math|<wide|x|\<vect\>>\<in\>\<bbb-R\><rsup|M>,<wide|y|\<vect\>>\<in\>\<bbb-R\><rsup|N>,<wide|z|\<vect\>>\<in\>\<bbb-R\><rsup|O>>
  follow the relationship

  <\equation*>
    <wide|x|\<vect\>>\<mapsto\><wide|y|\<vect\>>\<mapsto\><wide|z|\<vect\>>,
  </equation*>

  then the derivative of <math|<wide|z|\<vect\>>> with respect to
  <math|<wide|x|\<vect\>>> can be calculated as the following matrix
  multiplication

  <\equation>
    <wide*|<frac|d<wide|z|\<vect\>>|d<wide|x|\<vect\>>>|\<wide-underbrace\>><rsub|<around*|(|O,M|)>>=<wide*|<frac|d<wide|z|\<vect\>>|d<wide|y|\<vect\>>>|\<wide-underbrace\>><rsub|<around*|(|O,N|)>>
    <wide*|<frac|d<wide|y|\<vect\>>|d<wide|x|\<vect\>>>|\<wide-underbrace\>><rsub|<around*|(|N,M|)>><label|eq:vector-chain>,
  </equation>

  where the three matrices above from the left to right are the Jacobian of
  <math|<wide|z|\<vect\>>> with respect to <math|<wide|x|\<vect\>>>, the
  Jacobian of <math|<wide|z|\<vect\>>> with respect to
  <math|<wide|y|\<vect\>>>, and the Jacobian of <math|<wide|y|\<vect\>>> with
  respect to <math|<wide|x|\<vect\>>>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<matrix|<tformat|<table|<row|<cell|<frac|\<partial\>z<rsub|1>|\<partial\>x<rsub|1>>>|<cell|\<cdots\>>|<cell|<frac|\<partial\>z<rsub|1>|\<partial\>x<rsub|m>>>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|<frac|\<partial\>z<rsub|O>|\<partial\>x<rsub|1>>>|<cell|\<cdots\>>|<cell|<frac|\<partial\>z<rsub|O>|\<partial\>x<rsub|M>>>>>>>>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|<frac|\<partial\>z<rsub|1>|\<partial\>y<rsub|1>>>|<cell|\<cdots\>>|<cell|<frac|\<partial\>z<rsub|1>|\<partial\>y<rsub|N>>>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|<frac|\<partial\>z<rsub|O>|\<partial\>y<rsub|1>>>|<cell|\<cdots\>>|<cell|<frac|\<partial\>z<rsub|O>|\<partial\>y<rsub|N>>>>>>><matrix|<tformat|<table|<row|<cell|<frac|\<partial\>y<rsub|1>|\<partial\>x<rsub|1>>>|<cell|\<cdots\>>|<cell|<frac|\<partial\>y<rsub|1>|\<partial\>x<rsub|M>>>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|<frac|\<partial\>y<rsub|N>|\<partial\>x<rsub|1>>>|<cell|\<cdots\>>|<cell|<frac|\<partial\>y<rsub|N>|\<partial\>x<rsub|M>>>>>>>.>>>>
  </eqnarray*>

  To verify Equation <reference|eq:vector-chain>, one may start with the
  definition of matrix multiplication:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<around*|(|<frac|d<wide|z|\<vect\>>|d<wide|x|\<vect\>>>|)><rsub|i,j>>|<cell|=>|<cell|<big|sum><rsub|k=1><rsup|N><around*|(|<frac|d<wide|z|\<vect\>>|d<wide|y|\<vect\>>>|)><rsub|i,k><around*|(|<frac|\<partial\><wide|y|\<vect\>>|\<partial\><wide|x|\<vect\>>>|)><rsub|k,j>.>>>>
  </eqnarray*>

  But the left-hand side and right-hand side are, respectively, just\ 

  <\equation*>
    <frac|\<partial\>z<rsub|i>|\<partial\>x<rsub|j>>=<big|sum><rsub|k=1><rsup|N><frac|\<partial\>z<rsub|i>|\<partial\>y<rsub|k>>
    <frac|\<partial\>y<rsub|k>|\<partial\>x<rsub|j>>,
  </equation*>

  which is in essense no different from Equation <reference|eq:scalar-chain>!
  Also, note how this is the dot product of the <math|i>-th row of
  <math|d<wide|z|\<vect\>>/d<wide|y|\<vect\>>> and the <math|j>-th column of
  <math|d<wide|y|\<vect\>>/d<wide|x|\<vect\>>>.\ 

  <with|font-series|bold|Second generalization.> We can generalize the
  multivariate chain rule even further. Suppose (real) tensors, or
  multidimensional arrays, <math|X,Y,Z> follow the following relationship

  <\equation*>
    X\<rightarrow\>Y\<rightarrow\>Z.
  </equation*>

  This is the most general version, since tensors include scalars and
  vectors. Since these tensors can each have an arbitrary number of
  dimensions, we will resort to an example here, where we assume that
  <math|X\<in\>\<bbb-R\><rsup|A\<times\>B\<times\>C>,Y\<in\>\<bbb-R\><rsup|D\<times\>E>,Z\<in\>\<bbb-R\><rsup|F\<times\>G\<times\>H\<times\>I>>.
  The derivative of <math|Z> with respect to <math|X> can be written as

  <\equation>
    <wide*|<frac|d Z|d X>|\<wide-underbrace\>><rsub|<around*|(|<around*|(|F,G,H,I|)>,<around*|(|A,B,C|)>|)>>=<wide*|<frac|d
    Z|d Y>|\<wide-underbrace\>><rsub|<around*|(|<around*|(|F,G,H,I|)>,<around*|(|D,E|)>|)>>:
    <wide*|<frac|d Y|d X>|\<wide-underbrace\>><rsub|<around*|(|<around*|(|D,E|)>,<around*|(|A,B,C|)>|)>><label|eq:tensor-chain>
  </equation>

  where the three tensors are defined as

  <\eqnarray*>
    <tformat|<table|<row|<cell|<around*|(|<frac|d Z|d
    X>|)><rsub|<around*|(|f,g,h,i|)>,<around*|(|a,b,c|)>>>|<cell|=>|<cell|<frac|\<partial\>
    Z<rsub|f,g,h,i>|\<partial\> X<rsub|a,b,c>>>>|<row|<cell|<around*|(|<frac|d
    Z|d Y>|)><rsub|<around*|(|f,g,h,i|)>,<around*|(|d,e|)>>>|<cell|=>|<cell|<frac|\<partial\>
    Z<rsub|f,g,h,i>|\<partial\> Y<rsub|d,e>>>>|<row|<cell|<around*|(|<frac|d
    Y|d X>|)><rsub|<around*|(|d,e|)>,<around*|(|a,b,c|)>>>|<cell|=>|<cell|<frac|\<partial\>
    Y<rsub|d,e>|\<partial\> X<rsub|a,b,c>>>>>>
  </eqnarray*>

  and the \P:\Q operator denotes <with|font-shape|italic|tensor contraction>,
  which is defined as

  <\equation*>
    <around*|(|<frac|d Z|d X>|)><rsub|<around*|(|f,g,h,i|)>,<around*|(|a,b,c|)>>=<wide*|<around*|(|<frac|d
    Z|d Y>|)><rsub|<around*|(|f,g,h,i|)>,<around*|(|:,:|)>>|\<wide-underbrace\>><rsub|<around*|(|D,E|)>>\<cdot\>
    <wide*|<around*|(|<frac|d Y<rsub|>|d X>|)><rsub|<around*|(|:,:|)>,<around*|(|a,b,c|)>>|\<wide-underbrace\>><rsub|<around*|(|D,E|)>>=<frac|d
    Z<rsub|f,g,h,i>|d Y>\<cdot\><frac|d Y|d X<rsub|a,b,c>>.
  </equation*>

  The \P\<cdot\>\Q operator denotes the <with|font-shape|italic|tensor dot
  product>, which multiplies two tensors element-wise and sum up all
  resulting products into a single scalar. Unfortunately, tensors are
  high-dimensional so we can't write them out as in the vector case. But
  recall that the dot product was important in the vector case, too:
  <math|\<partial\>z<rsub|i>/\<partial\>x<rsub|j>> is the dot product of the
  <math|i>-th row of <math|d<wide|z|\<vect\>>/d<wide|y|\<vect\>>> and the
  <math|j>-th column of <math|d<wide|y|\<vect\>>/d<wide|x|\<vect\>>>.\ 

  <\remark>
    Equation <reference|eq:vector-chain> and <reference|eq:tensor-chain> are
    the same as Equation <reference|eq:scalar-chain> except that, for
    Equation <reference|eq:vector-chain> and <reference|eq:tensor-chain>, the
    variables are organized in more complicated structures. Therefore, one
    has to rely on matrix multiplication and, more generally, tensor
    contraction, to express the multivariate chain rule involving these
    complicated structures.
  </remark>

  <\eqnarray*>
    <tformat|<table|<row|<cell|f>|<cell|=>|<cell|m
    a<eq-number><label|eq:tensor-chain-sum>>>>>
  </eqnarray*>

  <section|The goal of automatic differentiation>

  Consider a directed acyclic computational graph that takes in <math|N>
  tensors and outputs <math|M> tensors (these tensors may be organized in
  pytrees<\footnote>
    \PIn JAX, we use the term pytree to refer to a tree-like structure built
    out of container-like Python objects.\Q
  </footnote>), with no restriction on the shape of each tensor. This is, of
  course, a very general setup. We view this graph as a function and we
  denote it by <math|g>:

  <\equation*>
    g<around*|(|X<rsup|<around*|(|1|)>>,\<ldots\>,X<rsup|<around*|(|N|)>>|)>=<around*|(|Y<rsup|<around*|(|1|)>>,\<ldots\>,Y<rsup|<around*|(|M|)>>|)>.
  </equation*>

  One main goal of automatic differentiation is to obtain the following
  matrix of tensors:\ 

  <\equation*>
    <text|\PJacobian\Q>=<matrix|<tformat|<table|<row|<cell|<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|1|)>>>>|<cell|\<cdots\>>|<cell|<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|N|)>>>>>|<row|<cell|\<vdots\>>|\<ddots\>|<cell|\<vdots\>>>|<row|<cell|<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|1|)>>>>|<cell|\<cdots\>>|<cell|<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|N|)>>>>>>>>,
  </equation*>

  where we have put quotes around Jacobian because this matrix above does
  indeed contain all the required partial derivatives but is not the standard
  Jacobian of vector-to-vector functions.\ 

  Since each tensor could in general have arbitrary shape, in Python, for
  example, the result would be a double nested tuple. Below, let's try out
  <verbatim|jax.jacfwd> and <verbatim|jax.jacrev>, two functions in JAX that
  compute the \PJacobian\Q, to see that this is indeed what happens.
  <verbatim|jax.jacfwd> and <verbatim|jax.jacrev> computes the \PJacobian\Q
  via <with|font-shape|italic|forward-mode> (Section <reference|sec:fm>) and
  <with|font-shape|italic|reverse-mode> automatic differentiation (Section
  <reference|sec:rm>) respectively. These two modes will be discussed at
  length in later sections.

  <\python-code>
    def g(X1, X2):

    \ \ \ \ Y1 = X1 @ X2

    \ \ \ \ Y2 = X * X

    \ \ \ \ return Y1, Y2

    \ \ \ \ 

    key = jax.random.key(42)

    X1_key, X2_key = jax.random.split(key, 2)

    X1 = jax.random.normal(X1_key, shape=(2, 3))

    X2 = jax.random.normal(X2_key, shape=(3, 4))

    \;

    Y1, Y2 = g(X1, X2)

    print(Y1.shape, Y2.shape) \ # (2, 4) (2, 3)

    \;

    jac_fwd = jax.jacfwd(g, argnums=(0, 1))(X1, X2)

    # first row of the nested tuple

    print(jac_fwd[0][0].shape, jac_fwd[0][1].shape) \ # (2, 4, 2, 3) (2, 4,
    3, 4)

    # second row of the nested tuple

    print(jac_fwd[1][0].shape, jac_fwd[1][1].shape) \ # (2, 3, 2, 3) (2, 3,
    3, 4)

    \;

    jac_rev = jax.jacrev(g, argnums=(0, 1))(X1, X2)

    # first row of the nested tuple

    print(jac_rev[0][0].shape, jac_rev[0][1].shape) \ # (2, 4, 2, 3), (2, 4,
    3, 4)

    # second row of the nested tuple

    print(jac_rev[1][0].shape, jac_rev[1][1].shape) \ # (2, 3, 2, 3), (2, 3,
    3, 4)
  </python-code>

  Before we start later sections, it's also important to note that, in
  practice, one would define the function <math|g> using some simpler
  functions or subroutines. Let <math|f> denote a function within the
  function <math|g> that takes in <math|N<rsub|f>> tensors and outputs
  <math|M<rsub|f>> tensors

  <\equation*>
    f<around*|(|X<rsup|<around*|(|<with|color|dark
    green|f>,1|)>>,\<ldots\>,X<rsup|<around*|(|<with|color|dark
    green|f>,N<rsub|f>|)>>|)>=<around*|(|Y<rsup|<around*|(|<with|color|dark
    green|f>,1|)>>,\<ldots\>,Y<rsup|<around*|(|<with|color|dark
    green|f>,M<rsub|f>|)>>|)>.
  </equation*>

  The outputs of <math|g> given its inputs can be computed by running
  Algorithm 1<\footnote>
    I must admit that the idea of looping through functions is a bit
    handwavy; the idea I'm trying to convey is just that we want to execute
    these functions in an order such that their required inputs are already
    computed.
  </footnote>:<\float|float|hf>
    <hrule>

    <\specified-algorithm>
      <with|font-series|bold|Forward pass through a computation graph>

      \;
    <|specified-algorithm>
      <with|font-series|bold|Input:> computation graph,
      (<math|X<rsup|<around*|(|1|)>>,\<ldots\>,X<rsup|<around*|(|N|)>>>)

      \;

      <with|font-series|bold|For> <math|f> in functions inside the
      computational graph (in a forward fashion):

      \;

      <space|2em><math|<around*|(|Y<rsup|<around*|(|f,1|)>>,\<ldots\>,Y<rsup|<around*|(|f,M<rsub|f>|)>>|)>\<leftarrow\>f<around*|(|X<rsup|<around*|(|f,1|)>>,\<ldots\>,X<rsup|<around*|(|f,N<rsub|f>|)>>|)>>

      \;

      <with|font-series|bold|Output:> (<math|Y<rsup|<around*|(|1|)>>,\<ldots\>,Y<rsup|<around*|(|M|)>>>)
    </specified-algorithm>
  </float>

  <section|Forward-mode automatic differentiation><label|sec:fm>

  In this section, we discuss forward-mode AD for computing the full
  \PJacobian\Q and a \PJacobian-vector product\Q (\PJVP\Q) of a computation
  graph.

  <subsection|Understanding <verbatim|jac.jacfwd>>

  If we apply Equation <reference|eq:tensor-chain-sum> to some
  <math|<with|color|dark green|f>> inside the computation graph <math|g>, we
  see that

  <\equation>
    <frac|\<partial\> Y<rsup|<around*|(|<with|color|dark
    green|f>,i|)>>|\<partial\>X<rsup|<around*|(|j|)>>>=<big|sum><rsub|k=1><rsup|N<rsub|f>><frac|\<partial\>
    Y<rsup|<around*|(|<with|color|dark green|f>,i|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>>:<frac|\<partial\> X<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>|\<partial\>X<rsup|<around*|(|j|)>>><space|1em><text|for
    all >i<text| and >j.<label|eq:tensor-chain-sum-rule-f>
  </equation>

  This relationship shows that, if we want to compute the partial derivatives
  of <math|f>'s output variables with respect to <math|g>'s input variables,
  we must first know the partial derivatives of <math|f>'s input variables
  with respsect to <math|g>'s input variables. As a result, it inspires a
  \Pforward-style\Q algorithm (i.e., the algorithm first goes through
  variables closer to <math|g>'s input variables) for computing the partial
  derivatives of <math|g>'s output variables with respect to <math|g>'s input
  variables (Algorithm <reference|algo:fmad>). <\float|float|fht>
    \;

    <\specified-algorithm>
      <with|font-series|bold|Forward-mode automatic differentiation for
      computing the \PJacobian\Q><label|algo:fmad>
    <|specified-algorithm>
      <with|font-series|bold|Input:> computation graph,
      <math|<around*|(|X<rsup|<around*|(|1|)>>,\<ldots\>,X<rsup|<around*|(|N|)>>|)>>

      \;

      <with|font-series|bold|Initialize> <math|<around*|(|<around*|(|<frac|\<partial\>X<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|1|)>>>,\<ldots\>,<frac|\<partial\>X<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|N|)>>>|)>,\<ldots\>,<around*|(|<frac|\<partial\>X<rsup|<around*|(|N|)>>|\<partial\>X<rsup|<around*|(|1|)>>>,\<ldots\>,<frac|\<partial\>X<rsup|<around*|(|N|)>>|\<partial\>X<rsup|<around*|(|N|)>>>|)>|)>>

      \;

      <with|font-series|bold|For> <math|f> in functions inside the
      computational graph (in a forward fashion):

      \;

      <space|2em><math|<around*|(|Y<rsup|<around*|(|f,1|)>>,\<ldots\>,Y<rsup|<around*|(|f,M<rsub|f>|)>>|)>\<leftarrow\>f<around*|(|X<rsup|<around*|(|f,1|)>>,\<ldots\>,X<rsup|<around*|(|f,N<rsub|f>|)>>|)>>

      \;

      <space|2em><math|<frac|\<partial\>Y<rsup|<around*|(|<with|color|dark
      green|f>,i|)>>|\<partial\>X<rsup|<around*|(|j|)>>>\<leftarrow\><big|sum><rsub|k=1><rsup|N<rsub|f>><frac|\<partial\>Y<rsup|<around*|(|<with|color|dark
      green|f>,i|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
      green|f>,k|)>>>:<frac|\<partial\>X<rsup|<around*|(|<with|color|dark
      green|f>,k|)>>|\<partial\>X<rsup|<around*|(|j|)>>><space|1em><text|for
      all >i<text| and >j>

      \;

      <with|font-series|bold|Output:> <math|<around*|(|<around*|(|<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|1|)>>>,\<ldots\>,<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|N|)>>>|)>,\<ldots\>,<around*|(|<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|1|)>>>,\<ldots\>,<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|N|)>>>|)>|)>>
    </specified-algorithm>
  </float>\ 

  Note that we need to have a forward pass within Algorithm
  <reference|algo:fmad> (line 4) because we need to evaluate each
  <math|><math|\<partial\> Y<rsup|<around*|(|i,f|)>>/\<partial\>X<rsup|<around*|(|k,f|)>>>
  at the actual value of <math|X<rsup|<around*|(|k,f|)>>>. Despite the
  correctness of Algorithm 2, it does not represent how JAX implements
  <verbatim|jax.jacfwd>. In Section <reference|sec:fwdjvp>, we derive the
  algorithm for computing the \PJVP\Q and show how it can be leveraged to
  compute the full \PJacobian\Q.

  <subsection|Understanding <verbatim|jac.jvp>><label|sec:fwdjvp>

  In certain scenarios, one doesn't need the full \PJacobian\Q, which
  contains the partial derivative of every output variable with respect to
  every input variable. Instead, one might only want the partial derivatives
  of all output variables with respect to one specific input variable, i.e.,
  a single scalar entry within the entire
  <math|<around*|(|X<rsup|<around*|(|1|)>>,\<ldots\>,X<rsup|<around*|(|N|)>>|)>>.
  Denoting this scalar input variable by <math|x\<in\>\<bbb-R\>>, we organize
  the desired partial derivatives as

  <\equation>
    <around*|(|<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>x>,\<ldots\>,<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>x>|)>,<label|exp:desired-pd>
  </equation>

  where <math|\<partial\>Y<rsup|<around*|(|i|)>>/\<partial\>x> has the same
  shape as <math|Y<rsup|<around*|(|i|)>>>.\ 

  We then make the important yet potentially non-trivial observation: we can
  obtain the quantity above using the following computation:

  <\equation>
    <around*|(|<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|1|)>>>:V<rsup|<around*|(|1|)>>+\<cdots\>+<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|N|)>>>:V<rsup|<around*|(|N|)>>,\<ldots\>,<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|1|)>>>:V<rsup|<around*|(|1|)>>+\<cdots\>+<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|N|)>>>:V<rsup|<around*|(|N|)>>|)>,<label|exp:jvp>
  </equation>

  where (a) <math|V<rsup|<around*|(|j|)>>> have the same shape as
  <math|X<rsup|<around*|(|j|)>>> and (b) every entry of
  <math|<around*|(|V<rsup|<around*|(|1|)>>,\<ldots\>,V<rsup|<around*|(|N|)>>|)>>
  is zero <with|font-shape|italic|except> the entry corresponding to
  <math|x>, the quantity we want to differentiate with respect to. This
  computation is called the \PJVP\Q<\footnote>
    To see where the name \PJacobian-vector product\Q comes from, consider
    the case in which a computation graph takes a single vector input
    <math|<wide|x|\<vect\>>> and outputs another vector
    <math|<wide|y|\<vect\>>>. In this case, the partial derivatives of all
    output variables with respect to a single input variable can be computed
    using

    <\equation*>
      <frac|\<partial\><wide|y|\<vect\>>|\<partial\>x<rsub|i>>=<frac|\<partial\><wide|y|\<vect\>>|\<partial\><wide|x|\<vect\>>>
      <wide|v|\<vect\>><space|1em>with<space|1em><wide|v|\<vect\>>=<wide|e|\<vect\>><rsub|i>,
    </equation*>

    which is, unanimously, the product of a Jacobian and a vector. Clearly,
    this naming convention was kept even when engineers and researchers
    generalized the input and output of a computation graph to be more than
    one and/or higher-dimensional.
  </footnote>. Let's verify that this is indeed what JAX's <verbatim|jax.jvp>
  computes:

  <\python-code>
    # let us choose x to be the (1, 2) entry of X1

    # below are three ways of obtaining the same answer in jax

    \;

    # first way

    # indexing the result of jacfwd (the jacobian)

    \;

    jac = jax.jacfwd(g, argnums=(0, 1))(X1, X2)

    jvp_via_indexing = (jac[0][0][:, :, 1, 2], jac[1][0][:, :, 1, 2])

    \;

    # second way

    # contracting the jacobian with V1 and V2

    \;

    V1, V2 = np.zeros(X1.shape), np.zeros(X2.shape)

    V1 = V1.at[1, 2].set(1)

    \;

    def contract(A, B):

    \ \ \ \ return np.einsum("ijkl,kl-\<gtr\>ij", A, B)

    \;

    jvp_after_jac_is_computed = (

    \ \ \ \ contract(jac[0][0], V1) + contract(jac[0][1], V2),\ 

    \ \ \ \ contract(jac[1][0], V1) + contract(jac[1][1], V2)

    )

    \;

    # third way

    # use jvp in jax

    \;

    primals, tangents = jax.jvp(g, (X1, X2), (V1, V2))

    \;

    print(np.allclose(jvp_via_indexing[0], tangents[0])) \ # true

    print(np.allclose(jvp_via_indexing[1], tangents[1])) \ # true

    print(np.allclose(jvp_after_jac_is_computed[0], tangents[0])) \ # true

    print(np.allclose(jvp_after_jac_is_computed[1], tangents[1])) \ # true
  </python-code>

  While we could first compute the \PJacobian\Q and then contract the
  matrices it contains with <math|V<rsup|<around*|(|1|)>>,\<ldots\>,V<rsup|<around*|(|M|)>>>
  (the second way in the code example above), it turns out to be much more
  efficient to compute the \PJVP\Q directly. Here's how this can be
  accomplished. Contract both sides of Equation
  <reference|eq:tensor-chain-sum-rule-f> with <math|V<rsup|<around*|(|j|)>>>,
  obtain

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>
    Y<rsup|<around*|(|i,f|)>>|\<partial\>X<rsup|<around*|(|j|)>>>:V<rsup|<around*|(|j|)>>>|<cell|=>|<cell|<around*|(|<big|sum><rsub|k=1><rsup|N<rsub|f>><frac|\<partial\>
    Y<rsup|<around*|(|i,f|)>>|\<partial\>X<rsup|<around*|(|k,f|)>>>:<frac|\<partial\>
    X<rsup|<around*|(|k,f|)>>|\<partial\>X<rsup|<around*|(|j|)>>>|)>:V<rsup|<around*|(|j|)>>.>>>>
  </eqnarray*>

  Since tensor contraction is distributive and commutative, we have

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|k=1><rsup|N<rsub|f>><around*|(|<frac|\<partial\>
    Y<rsup|<around*|(|i,f|)>>|\<partial\>X<rsup|<around*|(|k,f|)>>>:<frac|\<partial\>
    X<rsup|<around*|(|k,f|)>>|\<partial\>X<rsup|<around*|(|j|)>>>|)>:V<rsup|<around*|(|j|)>>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|k=1><rsup|N<rsub|f>><frac|\<partial\>
    Y<rsup|<around*|(|i,f|)>>|\<partial\>X<rsup|<around*|(|k,f|)>>>:<around*|(|<frac|\<partial\>
    X<rsup|<around*|(|k,f|)>>|\<partial\>X<rsup|<around*|(|j|)>>>:V<rsup|<around*|(|j|)>>|)>.>>>>
  </eqnarray*>

  Finally, summing across <math|j>, we have

  <\equation*>
    <wide*|<frac|\<partial\> Y<rsup|<around*|(|i,f|)>>|\<partial\>X<rsup|<around*|(|1|)>>>V<rsup|<around*|(|1|)>>+\<cdots\>+<frac|\<partial\>
    Y<rsup|<around*|(|i,f|)>>|\<partial\>X<rsup|<around*|(|N|)>>>V<rsup|<around*|(|N|)>>|\<wide-underbrace\>><rsub|\<triangleq\><wide|Y|\<dot\>><rsup|<around*|(|i,f|)>>>=<big|sum><rsub|k=1><rsup|N<rsub|f>><frac|\<partial\>
    Y<rsup|<around*|(|i,f|)>>|\<partial\>X<rsup|<around*|(|k,f|)>>>:<wide*|<around*|(|<frac|\<partial\>
    X<rsup|<around*|(|k,f|)>>|\<partial\>X<rsup|<around*|(|1|)>>>:V<rsup|<around*|(|1|)>>+\<cdots\>+<frac|\<partial\>
    X<rsup|<around*|(|k,f|)>>|\<partial\>X<rsup|<around*|(|N|)>>>:V<rsup|<around*|(|N|)>>|)>|\<wide-underbrace\>><rsub|\<triangleq\><wide|X|\<dot\>><rsup|<around*|(|k,f|)>>>,
  </equation*>

  where we have defined two new quantities
  <math|<wide|Y|\<dot\>><rsup|<around*|(|i,f|)>>> and
  <math|<wide|X|\<dot\>><rsup|<around*|(|k,f|)>>>. Again, using another
  \Pforward-style\Q algorithm (Algorithm <reference|algo:jvp>), we can
  eventually obtain <math|<wide|Y|\<dot\>><rsup|<around*|(|f|)>>>, which is
  exactly what we wanted at the beginning of this sub-section (Expression
  <reference|exp:desired-pd> and <reference|exp:jvp>).
  <float|float|t|<\specified-algorithm>
    <with|font-series|bold|Reverse-mode automatic differentiation for a
    \PJacobian-vector product\Q><label|algo:jvp>
  <|specified-algorithm>
    <with|font-series|bold|Input:> computation graph, primals
    <math|<around*|(|X<rsup|<around*|(|1|)>>,\<ldots\>,X<rsup|<around*|(|N|)>>|)>>,
    tangents <math|<around*|(|V<rsup|<around*|(|1|)>>,\<ldots\>,V<rsup|<around*|(|N|)>>|)>>

    <space|3em>

    Initialize <math|<around*|(|<wide|X|\<dot\>><rsup|<around*|(|1|)>>=V<rsup|<around*|(|1|)>>,\<ldots\>,<wide|X|\<dot\>><rsup|<around*|(|N|)>>=V<rsup|<around*|(|N|)>>|)>>

    \;

    <with|font-series|bold|For> <math|f> in functions inside the
    computational graph (in an appropriate order):

    \;

    <space|2em><math|Y<rsup|<around*|(|<with|color|dark
    green|f>,1|)>>,\<ldots\>Y<rsup|<around*|(|<with|color|dark
    green|f>,M<rsub|f>|)>>\<leftarrow\>f<around*|(|X<rsup|<around*|(|<with|color|dark
    green|f>,1|)>>,\<ldots\>X<rsup|<around*|(|<with|color|dark
    green|f>,N<rsub|f>|)>>|)>>

    \;

    <space|2em>For <math|i> in <math|1,\<ldots\>M>:

    \;

    <space|4em><math|<wide|Y|\<dot\>><rsup|<around*|(|<with|color|dark
    green|f>,i|)>>\<leftarrow\><big|sum><rsub|k=1><rsup|N<rsub|f>><frac|\<partial\>Y<rsup|<around*|(|<with|color|dark
    green|f>,i|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>><rsub|>:<wide|X|\<dot\>><rsup|<around*|(|f,k|)>>><space|1em>#
    line 6

    \;

    <with|font-series|bold|Output:> <math|<around*|(|<wide|Y|\<dot\>><rsup|<around*|(|1|)>><with|color|#a0a0a0|<around*|(|<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|1|)>>>:V<rsup|<around*|(|1|)>>+\<cdots\>+<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|N|)>>>:V<rsup|<around*|(|N|)>>|)>>,\<ldots\>,<wide|Y|\<dot\>><rsup|<around*|(|M|)>><with|color|#a0a0a0|<around*|(|<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|1|)>>>:V<rsup|<around*|(|1|)>>+\<cdots\>+<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|N|)>>>:V<rsup|<around*|(|N|)>>|)>>|)>>
  </specified-algorithm>>

  TODO: Obviously, not strictly required to ones and zeros (?), directional
  derivative and so on

  TODO: explain the words tangent and primals in the pushforward context

  <subsection|How JAX uses <verbatim|jac.jvp> to compute the \PJacobian\Q ?>

  Each time <verbatim|jac.jvp> allows us to compute the partial derivatives
  of all output variables with respect to a single input variable. This
  suggests that, we can simply call <verbatim|jac.jvp> once for ever input
  variable, and asemble the results from all the calls to the Jacobian.
  Indeed, this is what happens under the hood in JAX and here's how we might
  do it explicitly:

  <subsection|Defining custom JVP rules / pushforward rules via
  <verbatim|jax.custom_jvp> and <verbatim|f.defjvp>>

  Built-in functions of JAX certainly can be differentiated through, but what
  happens when incorporate code from another package into a computation
  graph. Can we still compute the JVP using JAX? Let's try it out

  <\python-code>
    # previously, this was g

    # def g(X1, X2):

    # \ \ \ Y1 = X1 @ X2

    # \ \ \ Y2 = X * X

    # \ \ \ return Y1, Y2

    \ \ \ \ 

    # suppose for some reason which we do not want to use matmul in jax but
    rather a custom matmul function (which uses numpy)

    # (jax.pure_callback is the way for including non-jax code in jax
    workflow)

    \;

    def matmul(A, B):

    \ \ \ \ result_shape = jax.core.ShapedArray((A.shape[0], B.shape[1]),
    A.dtype)

    \ \ \ \ return jax.pure_callback(npy.matmul, result_shape, A, B)

    \;

    def g2(A, B):

    \ \ \ \ C = matmul(A, B)

    \ \ \ \ D = A * A

    \ \ \ \ return C, D
  </python-code>

  If we try to naively use jax.jvp, we get into trouble

  <\python-code>
    primals, tangents = jax.jvp(g2, (X1, X2), (V1, V2))

    # ValueError: Pure callbacks do not support JVP. Please use
    `jax.custom_jvp` to use callbacks while taking gradients.
  </python-code>

  Why is this? Notice line 6 in Algorithm 3 is called the JVP rule for
  <math|f>, we must know how to do this for <math|f> yet doesn't know this.
  Defining the custom JVP rule (derived in Section XX):

  <\python-code>
    @jax.custom_jvp

    def matmul(A, B):

    \ \ \ \ result_shape = jax.core.ShapedArray((A.shape[0], B.shape[1]),
    A.dtype)

    \ \ \ \ return jax.pure_callback(npy.matmul, result_shape, A, B)

    \;

    @matmul.defjvp

    def matmul_jvp(primals, tangents):

    \ \ \ \ A, B = primals

    \ \ \ \ A_dot, B_dot = tangents

    \ \ \ \ primal_out = matmul(A, B)

    \ \ \ \ tangent_out = matmul(A_dot, B) + matmul(A, B_dot)

    \ \ \ \ return primal_out, tangent_out
  </python-code>

  Now jax.jvp does work and give the correct answer

  <\python-code>
    _, tangents_from_g = jax.jvp(g, (X1, X2), (V1, V2))

    _, tangents_from_g2 = jax.jvp(g2, (X1, X2), (V1, V2))

    print(np.allclose(tangents_from_g[0], tangents_from_g2[0])) \ # True

    print(np.allclose(tangents_from_g[1], tangents_from_g2[1])) \ # True
  </python-code>

  Of course, there are also other reasons why we might want to define the JVP
  rule by ourselves. Another use case is when functions are defined
  implicitly. I really hope to include an example of this.

  <\python-code>
    # TODO
  </python-code>

  TODO: mention how sometimes defining the JVP rule also allows you to do
  reverse mode (maybe give some examples after the next section?

  <section|Reverse-mode automatic differentiation><label|sec:rm>

  In this section, we discuss reverse-mode AD for computing the full
  \PJacobian\Q and a \Pvector-Jacobian product\Q (\PVJP\Q) of a computation
  graph.

  <subsection|Computing the full \PJacobian\Q>

  If we, again, apply Equation <reference|eq:tensor-chain-sum> to some
  <math|<with|color|dark green|f>> inside the computation graph <math|g>, we
  also see that

  <\equation>
    <frac|\<partial\> Y<rsup|<around*|(|i|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>>=<big|sum><rsub|k=1><rsup|M<rsub|f>><frac|\<partial\>
    Y<rsup|<around*|(|i|)>>|\<partial\>Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>>:<frac|\<partial\> Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>><space|1em><text|for all >i<text| and
    >j.<label|eq:tensor-chain-sum-rule-f-2>
  </equation>

  This relationship shows that, if we want to compute the partial derivatives
  of <math|g>'s output variables with respect to <math|f>'s input variables,
  we must first know the partial derivatives of <math|g>'s output variables
  with respsect to <math|f>'s output variables. As a result, it inspires a
  \Preverse-style\Q algorithm (i.e., the algorithm first goes through
  variables closer to <math|g>'s output variables) for computing the partial
  derivatives of <math|g>'s output variables with respect to <math|g>'s input
  variables (Algorithm <reference|algo:rmad>).
  <float|float|t|<\specified-algorithm>
    <with|font-series|bold|Reverse-mode automatic differentiation for
    computing the \PJacobian\Q><label|algo:rmad>
  <|specified-algorithm>
    <with|font-series|bold|Input:> computation graph,
    <math|<around*|(|X<rsup|<around*|(|1|)>>,\<ldots\>,X<rsup|<around*|(|N|)>>|)>>

    \;

    <with|font-series|bold|Initialize> <math|<around*|(|<around*|(|<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>Y<rsup|<around*|(|1|)>>>,\<ldots\>,<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>Y<rsup|<around*|(|M|)>>>|)>,\<ldots\>,<around*|(|<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>Y<rsup|<around*|(|1|)>>>,\<ldots\>,<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>Y<rsup|<around*|(|M|)>>>|)>|)>>

    \;

    Forward pass

    \;

    <with|font-series|bold|For> <math|f> in functions inside the
    computational graph (in a backward fashion):

    \;

    <space|2em><math|<frac|\<partial\> Y<rsup|<around*|(|i|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>>=<big|sum><rsub|k=1><rsup|M<rsub|f>><frac|\<partial\>
    Y<rsup|<around*|(|i|)>>|\<partial\>Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>>:<frac|\<partial\> Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>><space|1em><text|for all >i<text| and >j><space|1em># need
    stored values from forward pass

    \;

    <with|font-series|bold|Output:> <math|<around*|(|<around*|(|<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|1|)>>>,\<ldots\>,<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|N|)>>>|)>,\<ldots\>,<around*|(|<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|1|)>>>,\<ldots\>,<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|N|)>>>|)>|)>>
  </specified-algorithm>>

  Note that we need to perform a forward pass and store all intermediate
  values within Algorithm <reference|algo:rmad> before everything because we
  need to evaluate each <math|><math|\<partial\>
  Y<rsup|<around*|(|f,k|)>>/\<partial\>X<rsup|<around*|(|f,j|)>>> at the
  actual value of <math|X<rsup|<around*|(|f,j|)>>>. Despite the correctness
  of Algorithm <reference|algo:rmad>, it does not represent how JAX
  implements <verbatim|jax.jacrev>. In Section <reference|sec:jaxvjp>, we
  derive the algorithm for computing the \PVJP\Q and, in Section
  <reference|sec:jaxjacrev> show how it can be leveraged to compute the full
  \PJacobian\Q.

  <subsection|Understanding <verbatim|jax.vjp>><label|sec:jaxvjp>

  Instead of the full \PJacobian\Q, one might only want the partial
  derivatives of one specific output variable (i.e., a single scalar entry
  within the entire <math|<around*|(|X<rsup|<around*|(|1|)>>,\<ldots\>,X<rsup|<around*|(|M|)>>|)>>)
  with respect to all input variables. Denoting this scalar output variable
  by <math|y\<in\>\<bbb-R\>>, we organize the desired partial derivatives as

  <\equation>
    <around*|(|<frac|\<partial\>y|\<partial\>X<rsup|<around*|(|1|)>>>,\<ldots\>,<frac|\<partial\>y|\<partial\>X<rsup|<around*|(|N|)>>>|)>
  </equation>

  where <math|\<partial\>y/\<partial\>X<rsup|<around*|(|i|)>>> has the same
  shape as <math|X<rsup|<around*|(|i|)>>>. It turns out that we can obtain
  the quantity above using the following computation:

  <\equation>
    <around*|(|W<rsup|<around*|(|<with|color|orange|1>|)>>:<frac|\<partial\>Y<rsup|<around*|(|<with|color|orange|1>|)>>|\<partial\>X<rsup|<around*|(|1|)>>>+\<cdots\>+W<rsup|<around*|(|<with|color|orange|M>|)>>:<frac|\<partial\>Y<rsup|<around*|(|<with|color|orange|M>|)>>|\<partial\>X<rsup|<around*|(|1|)>>>,\<ldots\>,W<rsup|<around*|(|<with|color|orange|1>|)>>:<frac|\<partial\>Y<rsup|<around*|(|<with|color|orange|1>|)>>|\<partial\>X<rsup|<around*|(|N|)>>>+\<cdots\>+W<rsup|<around*|(|<with|color|orange|M>|)>>:<frac|\<partial\>Y<rsup|<around*|(|<with|color|orange|M>|)>>|\<partial\>X<rsup|<around*|(|N|)>>>|)>,
  </equation>

  where (a) <math|W<rsup|<around*|(|j|)>>> have the same shape as
  <math|X<rsup|<around*|(|j|)>>> and (b) every entry of
  <math|<around*|(|W<rsup|<around*|(|1|)>>,\<ldots\>,W<rsup|<around*|(|M|)>>|)>>
  is zero <with|font-shape|italic|except> the entry corresponding to
  <math|y>. This computation is called the \PVJP\Q<\footnote>
    To see where the name \Pvector-Jacobian product\Q comes from, consider
    the case in which a computation graph takes a single vector input
    <math|<wide|x|\<vect\>>> and outputs another vector
    <math|<wide|y|\<vect\>>>. In this case, the partial derivatives of a
    single output variable with respect to all input variables can be
    computed using

    <\equation*>
      <frac|\<partial\>y<rsub|i>|\<partial\><wide|x|\<vect\>>>=<wide|v|\<vect\>><rsup|T><frac|\<partial\><wide|y|\<vect\>>|\<partial\><wide|x|\<vect\>>>
      <space|1em>with<space|1em><wide|v|\<vect\>>=<wide|e|\<vect\>><rsub|i>,
    </equation*>

    which is, unanimously, the product of a vector (though transposed) and a
    Jacobian. Clearly, this naming convention was kept even when engineers
    and researchers generalized the input and output of a computation graph
    to be more than one and/or higher-dimensional.
  </footnote>. Let's verify that this is indeed what JAX's <verbatim|jax.vjp>
  computes:

  <\python-code>
    # make note of computation shapes here

    \;

    # let us choose y to be the (1, 2) entry of Y1

    # below are three ways of obtaining the same answer in jax

    \;

    # first way

    # indexing the result of jacrev (the jacobian)

    \;

    jac = jax.jacrev(g, argnums=(0, 1))(X1, X2)

    vjp_via_indexing = (jac[0][0][1, 2, :, :], jac[0][1][1, 2, :, :])

    \;

    # second way

    # contracting the jacobian with W1 and W2

    \;

    W1, W2 = np.zeros(Y1.shape), np.zeros(Y2.shape)

    W1 = W1.at[1, 2].set(1)

    \;

    def contract(A, B):

    \ \ \ \ return np.einsum("ij,ijkl-\<gtr\>kl", A, B)

    \;

    vjp_after_jac_is_computed = (

    \ \ \ \ contract(W1, jac[0][0]) + contract(W2, jac[1][0]),\ 

    \ \ \ \ contract(W1, jac[0][1]) + contract(W2, jac[1][1])

    )

    \;

    # third way

    # use vjp in jax

    \;

    primals, vjpfun = jax.vjp(g, X1, X2)

    cotangents = vjpfun((W1, W2))

    \;

    print(np.allclose(vjp_via_indexing[0], cotangents[0])) \ # true

    print(np.allclose(vjp_via_indexing[1], cotangents[1])) \ # true

    print(np.allclose(vjp_after_jac_is_computed[0], cotangents[0])) \ # true

    print(np.allclose(vjp_after_jac_is_computed[1], cotangents[1])) \ # true
  </python-code>

  While we could first compute the \PJacobian\Q and then contract the
  matrices it contains with <math|W<rsup|<around*|(|1|)>>,\<ldots\>,W<rsup|<around*|(|M|)>>>
  (the second way in the code example above), it turns out to be much more
  efficient to compute the \PVJP\Q directly. Here's how this can be
  accomplished. Pre-Contract both sides of Equation (should be 8 here) with
  <math|W<rsup|<around*|(|i|)>>>, obtain

  <\equation*>
    W<rsup|<around*|(|i|)>>:<frac|\<partial\>
    Y<rsup|<around*|(|i|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>>=W<rsup|<around*|(|i|)>>:<around*|(|<big|sum><rsub|k=1><rsup|M<rsub|f>><frac|\<partial\>
    Y<rsup|<around*|(|i|)>>|\<partial\>Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>>:<frac|\<partial\> Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>>|)>
  </equation*>

  Since tensor contraction is distributive and commutative, we have

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|k=1><rsup|M<rsub|f>>W<rsup|<around*|(|i|)>>:<around*|(|<frac|\<partial\>
    Y<rsup|<around*|(|i|)>>|\<partial\>Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>>:<frac|\<partial\> Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|k=1><rsup|M<rsub|f>><around*|(|W<rsup|<around*|(|i|)>>:<frac|\<partial\>
    Y<rsup|<around*|(|i|)>>|\<partial\>Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>>|)>:<frac|\<partial\> Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>>>>>>
  </eqnarray*>

  Finally, summing across <math|i>, we have

  <\equation*>
    <wide*|W<rsup|<around*|(|1|)>>:<frac|\<partial\>
    Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>>+\<cdots\>+W<rsup|<around*|(|M|)>>:<frac|\<partial\>
    Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>>|\<wide-underbrace\>><rsub|\<triangleq\><wide|Y|\<dot\>><rsup|<around*|(|i,f|)>>>=<big|sum><rsub|k=1><rsup|M<rsub|f>><wide*|<around*|(|W<rsup|<around*|(|1|)>>:<frac|\<partial\>
    Y<rsup|<around*|(|1|)>>|\<partial\>Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>>+\<cdots\>+W<rsup|<around*|(|M|)>>:<frac|\<partial\>
    Y<rsup|<around*|(|M|)>>|\<partial\>Y<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>>|)>|\<wide-underbrace\>><rsub|\<triangleq\><wide|X|\<dot\>><rsup|<around*|(|k,f|)>>>:<frac|\<partial\>
    Y<rsup|<around*|(|<with|color|dark green|f>,k|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,j|)>>>,
  </equation*>

  Stopped here last time: should we give these quantities new names?

  \;

  \;

  where we have defined two new quantities
  <math|<wide|Y|\<dot\>><rsup|<around*|(|i,f|)>>> and
  <math|<wide|X|\<dot\>><rsup|<around*|(|k,f|)>>>. Again, using another
  \Pforward-style\Q algorithm (Algorithm <reference|algo:jvp>), we can
  eventually obtain <math|<wide|Y|\<dot\>><rsup|<around*|(|f|)>>>, which is
  exactly what we wanted at the beginning of this sub-section (Expression
  <reference|exp:desired-pd> and <reference|exp:jvp>).
  <float|float|t|<\specified-algorithm>
    <with|font-series|bold|Reverse-mode automatic differentiation for a
    \Pvector-Jacobian product\Q><label|algo:vjp>
  <|specified-algorithm>
    <with|font-series|bold|Input:> computation graph, primals
    <math|<around*|(|X<rsup|<around*|(|1|)>>,\<ldots\>,X<rsup|<around*|(|N|)>>|)>>,
    tangents <math|<around*|(|V<rsup|<around*|(|1|)>>,\<ldots\>,V<rsup|<around*|(|N|)>>|)>>

    <space|3em>

    Initialize <math|<around*|(|<wide|X|\<dot\>><rsup|<around*|(|1|)>>=V<rsup|<around*|(|1|)>>,\<ldots\>,<wide|X|\<dot\>><rsup|<around*|(|N|)>>=V<rsup|<around*|(|N|)>>|)>>

    \;

    Forward pass

    \;

    <with|font-series|bold|For> <math|f> in functions inside the
    computational graph (in an appropriate order):

    \;

    <space|2em><math|Y<rsup|<around*|(|<with|color|dark
    green|f>,1|)>>,\<ldots\>Y<rsup|<around*|(|<with|color|dark
    green|f>,M<rsub|f>|)>>\<leftarrow\>f<around*|(|X<rsup|<around*|(|<with|color|dark
    green|f>,1|)>>,\<ldots\>X<rsup|<around*|(|<with|color|dark
    green|f>,N<rsub|f>|)>>|)>>

    \;

    <space|2em>For <math|i> in <math|1,\<ldots\>M>:

    \;

    <space|4em><math|<wide|Y|\<dot\>><rsup|<around*|(|<with|color|dark
    green|f>,i|)>>\<leftarrow\><big|sum><rsub|k=1><rsup|N<rsub|f>><frac|\<partial\>Y<rsup|<around*|(|<with|color|dark
    green|f>,i|)>>|\<partial\>X<rsup|<around*|(|<with|color|dark
    green|f>,k|)>>><rsub|>:<wide|X|\<dot\>><rsup|<around*|(|f,k|)>>><space|1em>#
    line 6

    \;

    <with|font-series|bold|Output:> <math|<around*|(|<wide|Y|\<dot\>><rsup|<around*|(|1|)>><with|color|#a0a0a0|<around*|(|<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|1|)>>>:V<rsup|<around*|(|1|)>>+\<cdots\>+<frac|\<partial\>Y<rsup|<around*|(|1|)>>|\<partial\>X<rsup|<around*|(|N|)>>>:V<rsup|<around*|(|N|)>>|)>>,\<ldots\>,<wide|Y|\<dot\>><rsup|<around*|(|M|)>><with|color|#a0a0a0|<around*|(|<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|1|)>>>:V<rsup|<around*|(|1|)>>+\<cdots\>+<frac|\<partial\>Y<rsup|<around*|(|M|)>>|\<partial\>X<rsup|<around*|(|N|)>>>:V<rsup|<around*|(|N|)>>|)>>|)>>
  </specified-algorithm>>

  TODO: Obviously, not strictly required to ones and zeros (?), directional
  derivative and so on

  TODO: explain the words tangent and primals in the pushforward context

  <subsection|Understanding <verbatim|jax.jacrev>><label|sec:jaxjacrev>

  \;

  <section|Comparing forward mode and reverse mode>

  \;

  <section|Derivations of some JVP / pushforward rules>

  <subsection|Scalar addition>

  <subsection|Scalar multiplication>

  <subsection|Scalar sine>

  <subsection|Broadcasted function>

  <subsection|Matrix-vector product>

  How can I adapt what I derived, to multiple inputs and outputs? Or when
  inputs and outputs are not vectors?

  <\equation*>
    J<rsub|<text|after >f><around*|(|<with|font-series|bold|x>|)><with|font-series|bold|v>=J<rsub|f><around*|(|<with|font-series|bold|x><rsup|<around*|(|f|)>>|)><wide|<with|font-series|bold|x>|\<dot\>><rsup|<around*|(|f|)>>
  </equation*>

  First case, multiple inputs:

  <\equation*>
    J<rsub|<text|after >f><around*|(|<with|font-series|bold|x>|)><with|font-series|bold|v>=J<rsub|f><around*|(|<with|font-series|bold|x><rsup|<around*|(|f|)>>|)><wide|<with|font-series|bold|x>|\<dot\>><rsup|<around*|(|f|)>>
  </equation*>

  We can simply break the Jacobian vector products into multiple pieces?
  Horizontal slices

  Second case, matrice inputs and matrix outputs

  <subsection|Scalar root-finding>

  <subsection|Matrix-matrix product>

  <\theorem>
    (JVP pushforward rule for matrix-matrix multiplication)

    Let function <math|f> represent matrix-matrix multiplication

    <\equation*>
      f<around*|(|A,B|)>=A B=C,
    </equation*>

    where <math|A\<in\>\<bbb-R\><rsup|n\<times\>m>,B\<in\>\<bbb-R\><rsup|m\<times\>o>,C\<in\>\<bbb-R\><rsup|n\<times\>o>>

    \;
  </theorem>

  \PJVP\Q rule:

  <\equation*>
    <wide*|<wide|C|\<dot\>>|\<wide-underbrace\>><rsub|<around*|(|N,O|)>>=<wide*|<frac|\<partial\>C|\<partial\>A>|\<wide-underbrace\>><rsub|<around*|(|<around*|(|N,O|)>,<around*|(|N,M|)>|)>>:<wide*|<wide|A|\<dot\>>|\<wide-underbrace\>><rsub|<around*|(|N,M|)>>+<wide*|<frac|\<partial\>C|\<partial\>B>|\<wide-underbrace\>><rsub|<around*|(|<around*|(|N,O|)>,<around*|(|M,O|)>|)>>:<wide*|<wide|B|\<dot\>>|\<wide-underbrace\>><rsub|<around*|(|M,O|)>>
  </equation*>

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|C|\<dot\>><rsub|i,j>>|<cell|=>|<cell|<big|sum><rsub|k,l><around*|(|<frac|\<partial\>C|\<partial\>A>|)><rsub|<around*|(|i,j|)>,<around*|(|k,l|)>><wide|A|\<dot\>><rsub|k,l>+<big|sum><rsub|k,l><around*|(|<frac|\<partial\>C|\<partial\>B>|)><rsub|<around*|(|i,j|)>,<around*|(|k,l|)>><wide|B|\<dot\>><rsub|k,l>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|k,l><frac|\<partial\>C<rsub|i,j>|\<partial\>A<rsub|k,l>>
    <wide|A|\<dot\>><rsub|k,l>+<big|sum><rsub|k,l><frac|\<partial\>C<rsub|i,j>|\<partial\>B<rsub|k,l>>
    <wide|B|\<dot\>><rsub|k,l>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|k,l><frac|\<partial\><around*|(|<big|sum><rsub|m>A<rsub|i,m>
    B<rsub|m,j>|)>|\<partial\>A<rsub|k,l>>
    <wide|A|\<dot\>><rsub|k,l>+<big|sum><rsub|k,l><frac|\<partial\><around*|(|<big|sum><rsub|m>A<rsub|i,m>
    B<rsub|m,j>|)>|\<partial\>B<rsub|k,l>>
    <wide|B|\<dot\>><rsub|k,l>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|k,l>\<delta\><rsub|i,k>
    B<rsub|l,j> <wide|A|\<dot\>><rsub|k,l>+<big|sum><rsub|k,l>\<delta\><rsub|l,j>
    A<rsub|i,k> <wide|B|\<dot\>><rsub|k,l>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|l>B<rsub|l,j>
    <wide|A|\<dot\>><rsub|i,l>+<big|sum><rsub|k>A<rsub|i,k>
    <wide|B|\<dot\>><rsub|k,j>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|l><wide|A|\<dot\>><rsub|i,l>
    B<rsub|l,j> +<big|sum><rsub|k>A<rsub|i,k> <wide|B|\<dot\>><rsub|k,j>>>>>
  </eqnarray*>

  Therefore

  <\equation*>
    <wide|C|\<dot\>>=<wide|A|\<dot\>>B+A<wide|B|\<dot\>>
  </equation*>

  <subsection|L2 loss>

  <subsection|Linear system>

  Let function <math|f> represent matrix-matrix multiplication

  <\equation*>
    f<around*|(|A,b|)>=<around*|{|<text|solve >A x=b<text| for
    >x|}>\<backassign\>x
  </equation*>

  <math|A\<in\>\<bbb-R\><rsup|N\<times\>N>,b\<in\>\<bbb-R\><rsup|N>,x\<in\>\<bbb-R\><rsup|N>>

  <\equation*>
    <wide|x|\<dot\>>=<frac|\<partial\> x|\<partial\>
    A>:<wide|A|\<dot\>>+<frac|\<partial\> x|\<partial\> b>:<wide|b|\<dot\>>
  </equation*>

  Implicit function

  \;

  <\eqnarray*>
    <tformat|<table|<row|<cell|<wide|x|\<dot\>><rsub|j>>|<cell|=>|<cell|<big|sum><rsub|k,l><around*|(|<frac|\<partial\>
    x|\<partial\> A>|)><rsub|j,<around*|(|k,l|)>>
    <wide|A|\<dot\>><rsub|k,l>+<big|sum><rsub|k><around*|(|<frac|\<partial\>
    x|\<partial\> b>|)><rsub|j,k><wide|b|\<dot\>><rsub|k>>>|<row|<cell|>|<cell|=>|<cell|<wide*|<big|sum><rsub|k,l><frac|\<partial\>
    x<rsub|j>|\<partial\> A<rsub|k l>> <wide|A|\<dot\>><rsub|k
    l>|\<wide-underbrace\>><rsub|\<triangleq\><wide|x|\<dot\>><rsub|i><rsup|<around*|[|A|]>>>+<wide*|<big|sum><rsub|j><frac|\<partial\>
    x<rsub|j>|\<partial\> b<rsub|k>> <wide|b|\<dot\>><rsub|k>|\<wide-underbrace\>><rsub|\<triangleq\><wide|x|\<dot\>><rsub|i><rsup|<around*|[|b|]>>>>>>>
  </eqnarray*>

  How does a specific <math|x> element relate to a specific <math|b> element?

  <\eqnarray*>
    <tformat|<table|<row|<cell|A x>|<cell|=>|<cell|b<rsub|>>>|<row|<cell|<big|sum><rsub|j>A<rsub|i,j><rsub|>
    x<rsub|j>>|<cell|=>|<cell|b<rsub|i>>>|<row|<cell|<around*|(|<big|sum><rsub|j>A<rsub|i,j><rsub|>
    x<rsub|j>|)>>|<cell|=>|<cell|<frac|\<partial\>|\<partial\>b<rsub|k>><around*|(|b<rsub|i>|)>>>|<row|<cell|<big|sum><rsub|j>A<rsub|i,j><rsub|>
    <frac|\<partial\>x<rsub|j>|\<partial\>b<rsub|k>>>|<cell|=>|<cell|\<delta\><rsub|i
    k>>>|<row|<cell|<big|sum><rsub|k><around*|(|<big|sum><rsub|j>A<rsub|i,j><rsub|>
    <frac|\<partial\>x<rsub|i>|\<partial\>b<rsub|k>>|)>
    <wide|b|\<dot\>><rsub|k>>|<cell|=>|<cell|<big|sum><rsub|k>\<delta\><rsub|i
    k><wide|b|\<dot\>><rsub|k><space|1em><around*|(|<text|dot
    product>|)>>>|<row|<cell|<big|sum><rsub|i>A<rsub|l,i><around*|(|<big|sum><rsub|j><rsub|>
    <frac|\<partial\>x<rsub|i>|\<partial\>b<rsub|j>>
    <wide|b|\<dot\>><rsub|j>|)>>|<cell|=>|<cell|<wide|b|\<dot\>><rsub|k>>>|<row|<cell|<big|sum><rsub|i>A<rsub|l,i>
    <wide|x|\<dot\>><rsub|i><rsup|<around*|[|b|]>>>|<cell|=>|<cell|<wide|b|\<dot\>><rsub|l>>>|<row|<cell|A<wide|x|\<dot\>><rsup|<around*|[|b|]>>>|<cell|=>|<cell|<wide|b|\<dot\>>>>>>
  </eqnarray*>

  which is also a linear system!

  <subsection|Nonlinear system solve>

  \;

  <subsection|Neural ODE>

  \;

  <subsection|Softmax>

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|frame-color|black>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|algo:fmad|<tuple|2|5>>
    <associate|algo:jvp|<tuple|3|7>>
    <associate|algo:rmad|<tuple|4|9>>
    <associate|algo:vjp|<tuple|5|11>>
    <associate|auto-1|<tuple|1|2>>
    <associate|auto-10|<tuple|4.2|9>>
    <associate|auto-11|<tuple|4.3|11>>
    <associate|auto-12|<tuple|5|11>>
    <associate|auto-13|<tuple|6|12>>
    <associate|auto-14|<tuple|6.1|12>>
    <associate|auto-15|<tuple|6.2|12>>
    <associate|auto-16|<tuple|6.3|12>>
    <associate|auto-17|<tuple|6.4|12>>
    <associate|auto-18|<tuple|6.5|12>>
    <associate|auto-19|<tuple|6.6|12>>
    <associate|auto-2|<tuple|2|3>>
    <associate|auto-20|<tuple|6.7|12>>
    <associate|auto-21|<tuple|6.8|13>>
    <associate|auto-22|<tuple|6.9|13>>
    <associate|auto-23|<tuple|6.10|13>>
    <associate|auto-24|<tuple|6.11|13>>
    <associate|auto-25|<tuple|6.12|13>>
    <associate|auto-3|<tuple|3|5>>
    <associate|auto-4|<tuple|3.1|5>>
    <associate|auto-5|<tuple|3.2|5>>
    <associate|auto-6|<tuple|3.3|7>>
    <associate|auto-7|<tuple|3.4|8>>
    <associate|auto-8|<tuple|4|9>>
    <associate|auto-9|<tuple|4.1|9>>
    <associate|eq:scalar-chain|<tuple|1|2>>
    <associate|eq:tensor-chain|<tuple|3|3>>
    <associate|eq:tensor-chain-sum|<tuple|4|3>>
    <associate|eq:tensor-chain-sum-rule-f|<tuple|5|5>>
    <associate|eq:tensor-chain-sum-rule-f-2|<tuple|8|9>>
    <associate|eq:vector-chain|<tuple|2|2>>
    <associate|exp:desired-pd|<tuple|6|5>>
    <associate|exp:jvp|<tuple|7|6>>
    <associate|footnote-1|<tuple|1|1>>
    <associate|footnote-2|<tuple|2|1>>
    <associate|footnote-3|<tuple|3|1>>
    <associate|footnote-4|<tuple|4|1>>
    <associate|footnote-5|<tuple|5|3>>
    <associate|footnote-6|<tuple|6|4>>
    <associate|footnote-7|<tuple|7|6>>
    <associate|footnote-8|<tuple|8|10>>
    <associate|footnr-1|<tuple|1|1>>
    <associate|footnr-2|<tuple|2|1>>
    <associate|footnr-3|<tuple|3|1>>
    <associate|footnr-4|<tuple|4|1>>
    <associate|footnr-5|<tuple|5|3>>
    <associate|footnr-6|<tuple|6|4>>
    <associate|footnr-7|<tuple|7|6>>
    <associate|footnr-8|<tuple|8|10>>
    <associate|sec:fm|<tuple|3|5>>
    <associate|sec:fwdjvp|<tuple|3.2|5>>
    <associate|sec:jaxjacrev|<tuple|4.3|11>>
    <associate|sec:jaxvjp|<tuple|4.2|9>>
    <associate|sec:rm|<tuple|4|9>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Multivariate
      chain rule> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>The
      goal of automatic differentiation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Forward-mode
      automatic differentiation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <with|par-left|<quote|1tab>|3.1<space|2spc>Understanding
      <with|font-family|<quote|tt>|language|<quote|verbatim>|jac.jacfwd>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|3.2<space|2spc>Understanding
      <with|font-family|<quote|tt>|language|<quote|verbatim>|jac.jvp>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|3.3<space|2spc>How JAX uses
      <with|font-family|<quote|tt>|language|<quote|verbatim>|jac.jvp> to
      compute the \PJacobian\Q ? <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|3.4<space|2spc>Defining custom JVP rules /
      pushforward rules via <with|font-family|<quote|tt>|language|<quote|verbatim>|jax.custom_jvp>
      and <with|font-family|<quote|tt>|language|<quote|verbatim>|f.defjvp>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Reverse-mode
      automatic differentiation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>

      <with|par-left|<quote|1tab>|4.1<space|2spc>Computing the full
      \PJacobian\Q <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9>>

      <with|par-left|<quote|1tab>|4.2<space|2spc>Understanding
      <with|font-family|<quote|tt>|language|<quote|verbatim>|jax.vjp>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10>>

      <with|par-left|<quote|1tab>|4.3<space|2spc>Understanding
      <with|font-family|<quote|tt>|language|<quote|verbatim>|jax.jacrev>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Comparing
      forward mode and reverse mode> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-12><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|6<space|2spc>Derivations
      of some JVP / pushforward rules> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-13><vspace|0.5fn>

      <with|par-left|<quote|1tab>|6.1<space|2spc>Scalar addition
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-14>>

      <with|par-left|<quote|1tab>|6.2<space|2spc>Scalar multiplication
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-15>>

      <with|par-left|<quote|1tab>|6.3<space|2spc>Scalar sine
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-16>>

      <with|par-left|<quote|1tab>|6.4<space|2spc>Broadcasted function
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-17>>

      <with|par-left|<quote|1tab>|6.5<space|2spc>Matrix-vector product
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-18>>

      <with|par-left|<quote|1tab>|6.6<space|2spc>Scalar root-finding
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-19>>

      <with|par-left|<quote|1tab>|6.7<space|2spc>Matrix-matrix product
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-20>>

      <with|par-left|<quote|1tab>|6.8<space|2spc>L2 loss
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-21>>

      <with|par-left|<quote|1tab>|6.9<space|2spc>Linear system
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-22>>

      <with|par-left|<quote|1tab>|6.10<space|2spc>Nonlinear system solve
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-23>>

      <with|par-left|<quote|1tab>|6.11<space|2spc>Neural ODE
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-24>>

      <with|par-left|<quote|1tab>|6.12<space|2spc>Softmax
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-25>>
    </associate>
  </collection>
</auxiliary>