<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Notes on Multinomial Logistic
  Regression>|<doc-author|<author-data|<author-name|Zhihan
  Yang>>>|<doc-date|Feb 28, 2024>>

  <\table-of-contents|toc>
    <vspace*|1fn><with|font-series|bold|math-font-series|bold|1<space|2spc>Introduction>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-1><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|2<space|2spc>Model
    definition> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-2><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|3<space|2spc>Derivative
    with respect to <with|mode|math|vec<around*|(|W|)>>>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-3><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|4<space|2spc>Hessian
    with respect to <with|mode|math|vec<around*|(|W|)>>>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-4><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|5<space|2spc>Derivative
    with respect to <with|mode|math|<with|font-series|bold|W>>>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-5><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|6<space|2spc>Comparing
    results against JAX autodiff> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-6><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|Bibliography>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-7><vspace|0.5fn>
  </table-of-contents>

  <section|Introduction>

  What's the value of manual derivation (other than learning math)?

  <section|Model definition>

  \;

  <section|Derivative with respect to <math|vec<around*|(|W|)>>>

  The loss function for multinomial logistic regression is

  <\equation*>
    f<around*|(|<with|font-series|bold|W>|)>=-\<ell\><around*|(|<with|font-series|bold|W>|)>=<big|sum><rsub|i=1><rsup|N><around*|[|log<around*|(|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|)>-<around*|(|<big|sum><rsub|c=1><rsup|C>y<rsub|i
    c><with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|]>.
  </equation*>

  <subsection|Scalar Calculus>

  Following Exercise 8.4:

  <\equation*>
    \<mu\><rsub|i k>=<with|font|cal|S><around*|(|<with|font-series|bold|\<eta\>><rsub|i>|)><rsub|k>=<frac|exp<around*|(|\<eta\><rsub|i
    k>|)>|<big|sum><rsub|k<rprime|'>=1>exp<around*|(|\<eta\><rsub|i
    k<rprime|'>>|)>>
  </equation*>

  Using quotient rule:

  <\eqnarray*>
    <tformat|<table|<row|<cell|u>|<cell|=>|<cell|exp<around*|(|\<eta\><rsub|i
    k>|)>>>|<row|<cell|u<rprime|'>>|<cell|=>|<cell|\<delta\><rsub|k
    j>exp<around*|(|\<eta\><rsub|i k>|)><space|1em><around*|(|<text|could be
    <math|\<delta\><rsub|k j>exp<around*|(|\<eta\><rsub|i j>|)>> as
    well>|)>>>|<row|<cell|v>|<cell|=>|<cell|<big|sum><rsub|k<rprime|'>=1>exp<around*|(|\<eta\><rsub|i
    k<rprime|'>>|)>>>|<row|<cell|v<rprime|'>>|<cell|=>|<cell|exp<around*|(|\<eta\><rsub|i
    j>|)>>>>>
  </eqnarray*>

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>\<mu\><rsub|i
    k>|\<partial\>\<eta\><rsub|i j>>>|<cell|=>|<cell|<frac|\<delta\><rsub|k
    j>exp<around*|(|\<eta\><rsub|i k>|)><around*|(|<big|sum><rsub|k<rprime|'>=1>exp<around*|(|\<eta\><rsub|i
    k<rprime|'>>|)>|)>-exp<around*|(|\<eta\><rsub|i
    k>|)>exp<around*|(|\<eta\><rsub|i j>|)>|<around*|(|<big|sum><rsub|k<rprime|'>=1>exp<around*|(|\<eta\><rsub|i
    k<rprime|'>>|)>|)><rsup|2>>>>|<row|<cell|>|<cell|=>|<cell|\<delta\><rsub|k
    j>\<times\><frac|exp<around*|(|\<eta\><rsub|i
    k>|)>|<big|sum><rsub|k<rprime|'>=1>exp<around*|(|\<eta\><rsub|i
    k<rprime|'>>|)>>-<frac|exp<around*|(|\<eta\><rsub|i
    k>|)>|<big|sum><rsub|k<rprime|'>=1>exp<around*|(|\<eta\><rsub|i
    k<rprime|'>>|)>>\<times\><frac|exp<around*|(|\<eta\><rsub|i
    j>|)>|<big|sum><rsub|k<rprime|'>=1>exp<around*|(|\<eta\><rsub|i
    k<rprime|'>>|)>>>>|<row|<cell|>|<cell|=>|<cell|\<delta\><rsub|k
    j>\<mu\><rsub|i k>-\<mu\><rsub|i k>\<mu\><rsub|i
    j>>>|<row|<cell|=>|<cell|=>|<cell|\<mu\><rsub|i
    k><around*|(|\<delta\><rsub|k j>-\<mu\><rsub|i j>|)>>>>>
  </eqnarray*>

  The logic diagram:

  \;

  \;

  \;

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>l|\<partial\>w<rsub|c
    d>>>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|[|log<around*|(|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|)>-<around*|(|<big|sum><rsub|c=1><rsup|C>y<rsub|i
    c><with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|]>>>>>
  </eqnarray*>

  <subsection|Vector Calculus>

  We want to derive the gradient and Hessian of <math|f> with respect
  <math|<with|font-series|bold|W>>. While it's possible to derive
  <math|\<partial\>f/\<partial\><with|font-series|bold|W>> by hand (see
  Section <reference|matrix-derivation>), it's very inconvenient to derive
  the Hessian by hand because it is four-dimensional. Instead, we define
  <math|<with|font-series|bold|w>=vec<around*|(|<with|font-series|bold|W>|)>>
  (i.e., stacking the columns of <math|<with|font-series|bold|W>>) and
  consider

  <\equation*>
    \<nabla\>f<around*|(|<with|font-series|bold|w>|)>\<triangleq\><frac|\<partial\>f|\<partial\><with|font-series|bold|w>><infix-and>\<nabla\><rsup|2>f<around*|(|<with|font-series|bold|w>|)>\<triangleq\><frac|\<partial\>f|<around*|(|\<partial\><with|font-series|bold|w>|)><around*|(|\<partial\><with|font-series|bold|w>|)><rsup|T>>.
  </equation*>

  But it's also inconvenient to derive the entire
  <math|\<nabla\>f<around*|(|<with|font-series|bold|w>|)>> at once. To see
  this, try writing <math|f> in terms of <math|<with|font-series|bold|w>>
  without using <math|<with|font-series|bold|w><rsub|c>>'s; it's hard!
  Instead, we recognize that <math|\<nabla\>f<around*|(|<with|font-series|bold|w>|)>>
  simply stacks <math|\<nabla\><rsub|<with|font-series|bold|w><rsub|c>>f<around*|(|<with|font-series|bold|w>|)>>'s
  so we can just derive <math|\<nabla\><rsub|<with|font-series|bold|w><rsub|c>>f<around*|(|<with|font-series|bold|w>|)>>'s
  and stack them up:

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|\<partial\>\<ell\>|\<partial\><with|font-series|bold|w><rsub|c<rprime|'>>>>|<cell|=>|<cell|<frac|\<partial\>|\<partial\><with|font-series|bold|w><rsub|c<rprime|'>>><around*|{|<big|sum><rsub|i=1><rsup|N><around*|[|log<around*|(|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|)>-<around*|(|<big|sum><rsub|c=1><rsup|C>y<rsub|i
    c><with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|]>|}>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|[|<frac|\<partial\>|\<partial\><with|font-series|bold|w><rsub|c<rprime|'>>><around*|{|log<around*|(|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|)>|}>-<frac|\<partial\>|\<partial\><with|font-series|bold|w><rsub|c<rprime|'>>><around*|{|<around*|(|<big|sum><rsub|c=1><rsup|C>y<rsub|i
    c><with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|}>|]>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|[|<frac|1|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>>\<times\><frac|\<partial\>|\<partial\><with|font-series|bold|w><rsub|c<rprime|'>>><around*|{|exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|'>><with|font-series|bold|x><rsub|i>|)>|}>-<frac|\<partial\>|\<partial\><with|font-series|bold|w><rsub|c<rprime|'>>><around*|{|y<rsub|i
    c><with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|}>|]>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|[|<frac|<around*|(|exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|'>><with|font-series|bold|x><rsub|i>|)>|)>|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>><with|font-series|bold|x><rsub|i>-y<rsub|i
    c<rprime|'>><with|font-series|bold|x><rsub|i>|]>.>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|(|\<mu\><rsub|i
    c<rprime|'>>-y<rsub|i c<rprime|'>>|)><with|font-series|bold|x><rsub|i>.>>>>
  </eqnarray*>

  Therefore, we have

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|<frac|\<partial\>\<ell\>|\<partial\><with|font-series|bold|w>>>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|(|<with|font-series|bold|\<mu\>><rsub|i
    >-<with|font-series|bold|y><rsub|i >|)>\<otimes\><with|font-series|bold|x><rsub|i>>>>>>,
  </equation*>

  where <math|\<otimes\>> denotes the Kronecker product.

  <section|Hessian with respect to <math|vec<around*|(|W|)>>>

  <subsection|Scalar calculus>

  \;

  <subsection|Vector calculus>

  Here we derive <math|\<nabla\><rsup|2>f<around*|(|<with|font-series|bold|w>|)>>
  by computing the differential of <math|<with|font-series|bold|g><rsub|c<rprime|'>>>
  (tODO: definition) (with respect of <math|<with|font-series|bold|w><rsub|c<rprime|''>>>)
  <cite|minka2000old>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<partial\><with|font-series|bold|g><rsub|c<rprime|'>>>|<cell|=>|<cell|\<partial\><around*|{|<big|sum><rsub|i=1><rsup|N><around*|[|<frac|<around*|(|exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|'>><with|font-series|bold|x><rsub|i>|)>|)>|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>>-y<rsub|i
    c<rprime|'>>|]><with|font-series|bold|x><rsub|i>|}>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N>\<partial\><around*|{|<frac|<around*|(|exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|'>><with|font-series|bold|x><rsub|i>|)>|)>|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>>-y<rsub|i
    c<rprime|'>>|}><with|font-series|bold|x><rsub|i>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N>\<partial\><around*|{|<frac|<around*|(|exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|'>><with|font-series|bold|x><rsub|i>|)>|)>|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>>|}><with|font-series|bold|x><rsub|i><space|1em><around*|(|y<rsub|i
    c<rprime|'>><text| is independent of ><with|font-series|bold|w><rsub|c>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|[|<frac|\<delta\><rsub|c<rprime|'>,c<rprime|''>>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|''>><with|font-series|bold|x><rsub|i>|)><around*|(|d<with|font-series|bold|w><rsub|c<rprime|''>>|)><rsup|T><with|font-series|bold|x><rsub|i><around*|(|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|)>-<around*|(|exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|'>><with|font-series|bold|x><rsub|i>|)>|)>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|''>><with|font-series|bold|x><rsub|i>|)><around*|(|d<with|font-series|bold|w><rsub|c<rprime|''>>|)><rsup|T><with|font-series|bold|x><rsub|i>|<around*|(|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|)><rsup|2>>|]><with|font-series|bold|x><rsub|i>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|[|<frac|\<delta\><rsub|c<rprime|'>,c<rprime|''>>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|''>><with|font-series|bold|x><rsub|i>|)><around*|(|d<with|font-series|bold|w><rsub|c<rprime|''>>|)><rsup|T><with|font-series|bold|x><rsub|i><around*|(|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|)>|<around*|(|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|)><rsup|2>>-<frac|exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|'>><with|font-series|bold|x><rsub|i>|)>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|''>><with|font-series|bold|x><rsub|i>|)><around*|(|d<with|font-series|bold|w><rsub|c<rprime|''>>|)><rsup|T><with|font-series|bold|x><rsub|i>|<around*|(|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>|)><rsup|2>>|]><with|font-series|bold|x><rsub|i>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|[|\<delta\><rsub|c<rprime|'>,c<rprime|''>>
    <frac|exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|''>><with|font-series|bold|x><rsub|i>|)>|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>>
    <around*|(|d<with|font-series|bold|w><rsub|c<rprime|''>>|)><rsup|T><with|font-series|bold|x><rsub|i>-<frac|exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|'>><with|font-series|bold|x><rsub|i>|)>|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>>
    <frac|exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c<rprime|''>><with|font-series|bold|x><rsub|i>|)>|<big|sum><rsub|c=1><rsup|C>exp<around*|(|<with|font-series|bold|w><rsup|T><rsub|c><with|font-series|bold|x><rsub|i>|)>>
    <around*|(|d<with|font-series|bold|w><rsub|c<rprime|''>>|)><rsup|T><with|font-series|bold|x><rsub|i>|]><with|font-series|bold|x><rsub|i>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|[|\<delta\><rsub|c<rprime|'>,c<rprime|''>>
    \<mu\><rsub|i c<rprime|''>> <with|font-series|bold|x><rsub|i><rsup|T><around*|(|d<with|font-series|bold|w><rsub|c<rprime|''>>|)><rsup|>-\<mu\><rsub|i
    c<rprime|'>> \<mu\><rsub|i c<rprime|''>>
    <with|font-series|bold|x><rsub|i><rsup|T><around*|(|d<with|font-series|bold|w><rsub|c<rprime|''>>|)><rsup|>|]><with|font-series|bold|x><rsub|i>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|[|\<delta\><rsub|c<rprime|'>,c<rprime|''>>
    \<mu\><rsub|i c<rprime|''>> <with|font-series|bold|x><rsub|i><with|font-series|bold|x><rsub|i><rsup|T><around*|(|d<with|font-series|bold|w><rsub|c<rprime|''>>|)><rsup|>-\<mu\><rsub|i
    c<rprime|'>> \<mu\><rsub|i c<rprime|''>><with|font-series|bold|x><rsub|i>
    <with|font-series|bold|x><rsub|i><rsup|T><around*|(|d<with|font-series|bold|w><rsub|c<rprime|''>>|)><rsup|>|]>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|(|\<delta\><rsub|c<rprime|'>,c<rprime|''>>
    \<mu\><rsub|i c<rprime|''>><with|font-series|bold|x><rsub|i><rsup|>
    <with|font-series|bold|x><rsub|i><rsup|T>-\<mu\><rsub|i c<rprime|'>>
    \<mu\><rsub|i c<rprime|''>>|)>d<with|font-series|bold|w><rsub|c<rprime|''>><rsup|>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|<big|sum><rsub|i=1><rsup|N><around*|(|\<delta\><rsub|c<rprime|'>,c<rprime|''>>
    \<mu\><rsub|i c<rprime|''>><with|font-series|bold|x><rsub|i><rsup|>
    <with|font-series|bold|x><rsub|i><rsup|T>-\<mu\><rsub|i c<rprime|'>>
    \<mu\><rsub|i c<rprime|''>>|)>|)>d<with|font-series|bold|w><rsub|c<rprime|''>><rsup|>>>>>
  </eqnarray*>

  which is one of the six concanical forms mentioned in <cite|minka2000old>;
  from it, we can read off that the derivative is:

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|<frac|\<partial\><with|font-series|bold|g><rsub|c<rprime|'>>|\<partial\><with|font-series|bold|w><rsub|c<rprime|''>>>>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|N><around*|(|\<delta\><rsub|c<rprime|'>,c<rprime|''>>
    \<mu\><rsub|i c<rprime|''>><with|font-series|bold|x><rsub|i><rsup|>
    <with|font-series|bold|x><rsub|i><rsup|T>-\<mu\><rsub|i c<rprime|'>>
    \<mu\><rsub|i c<rprime|''>>|)>>>>>>
  </equation*>

  Getting the complete Hessian:

  <section|Derivative with respect to <math|<with|font-series|bold|W>>>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|\<partial\><around*|(|sum<around*|(|ln<around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)><with|font-series|bold|1>|)>|)>|)>>>|<row|<cell|>|<cell|=>|<cell|<around*|\<nobracket\>|sum<around*|(|\<partial\>
    ln<around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)><with|font-series|bold|1>|)>|)>|)>>>|<row|<cell|>|<cell|=>|<cell|sum<around*|(|<around*|(|vec<around*|(|1|)>\<oslash\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)><with|font-series|bold|1>|)>|\<nobracket\>>\<odot\>\<partial\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)><with|font-series|bold|1>|)>|)>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|vec<around*|(|1|)>\<oslash\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)><with|font-series|bold|1>|)>|)><rsup|T>
    \<partial\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)><with|font-series|bold|1>|)>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|vec<around*|(|1|)>\<oslash\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)><with|font-series|bold|1>|)>|)><rsup|T>
    \<partial\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)>|)><with|font-series|bold|1>>>|<row|<cell|>|<cell|=>|<cell|<around*|(|vec<around*|(|1|)>\<oslash\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)><with|font-series|bold|1>|)>|)><rsup|T>
    <around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)>\<odot\><around*|(|<with|font-series|bold|X><around*|(|d<with|font-series|bold|W>|)><rsup|T>|)>|)><with|font-series|bold|1>>>|<row|<cell|>|<cell|=>|<cell|Tr<around*|<left|[|3>|<with|font-series|bold|1><rsup|T>diag<around*|(|vec<around*|(|1|)>\<oslash\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)>\<cdot\><with|font-series|bold|1>|)>|)><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)>\<odot\><around*|(|<around*|(|d<with|font-series|bold|X>|)><with|font-series|bold|W><rsup|T>|)>|)><with|font-series|bold|1>|<right|]|3>>>>|<row|<cell|>|<cell|=>|<cell|Tr<around*|<left|[|3>|diag<around*|(|vec<around*|(|1|)>\<oslash\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)>\<cdot\><with|font-series|bold|1>|)>|)><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)>\<odot\><around*|(|<with|font-series|bold|X><around*|(|d<with|font-series|bold|W>|)><rsup|T>|)>|)><with|font-series|bold|1><with|font-series|bold|1><rsup|T>|<right|]|3>>>>|<row|<cell|>|<cell|=>|<cell|Tr<around*|<left|[|3>|diag<around*|(|vec<around*|(|1|)>\<oslash\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)>\<cdot\><with|font-series|bold|1>|)>|)><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)><around*|(|d<with|font-series|bold|W>|)><with|font-series|bold|X><with|font-series|bold|><rsup|T>|)>|<right|]|3>><space|1em><around*|(|\<ast\>|)>>>|<row|<cell|>|<cell|=>|<cell|Tr<around*|<left|[|3>|<with|font-series|bold|X><with|font-series|bold|><rsup|T>diag<around*|(|vec<around*|(|1|)>\<oslash\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)>\<cdot\><with|font-series|bold|1>|)>|)>exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)><around*|(|d<with|font-series|bold|W>|)>|<right|]|3>>>>|<row|<cell|>|<cell|>|<cell|exp<around*|(|<with|font-series|bold|><with|font-series|bold|W><with|font-series|bold|X><rsup|T>|)>
    diag<around*|(|vec<around*|(|1|)>\<oslash\><around*|(|exp<around*|(|<with|font-series|bold|X><with|font-series|bold|W><rsup|T>|)>\<cdot\><with|font-series|bold|1>|)>|)><with|font-series|bold|X><with|font-series|bold|><rsup|T>>>|<row|<cell|>|<cell|>|<cell|exp<around*|(|<with|font-series|bold|><with|font-series|bold|W><with|font-series|bold|X><rsup|T>|)>
    diag<around*|(|vec<around*|(|1|)><rsup|T>\<oslash\><around*|(|<with|font-series|bold|1><rsup|T>\<cdot\>exp<around*|(|<with|font-series|bold|W><with|font-series|bold|X><rsup|T>|)>|)>|)><with|font-series|bold|X><with|font-series|bold|><rsup|T>>>>>
  </eqnarray*>

  \;

  \;

  <\lemma>
    Let <math|A\<in\>\<bbb-R\><rsup|m\<times\>n>>,
    <math|B\<in\>\<bbb-R\><rsup|m\<times\>n>> and
    <math|c\<in\>\<bbb-R\><rsup|m>> Then, we have

    <\eqnarray*>
      <tformat|<table|<row|<cell|Tr<around*|(|diag<around*|(|c|)>\<cdot\><around*|(|A\<odot\>B|)>\<cdot\><with|font-series|bold|1><rsub|n\<times\>m>|)>>|<cell|=>|<cell|Tr<around*|(|diag<around*|(|c|)>\<cdot\>A\<cdot\>B<rsup|T>|)>.>>>>
    </eqnarray*>
  </lemma>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|<matrix|<tformat|<table|<row|<cell|c<rsub|1>>|<cell|>|<cell|0>>|<row|<cell|>|<cell|\<ddots\>>|<cell|>>|<row|<cell|0>|<cell|>|<cell|c<rsub|m>>>>>><matrix|<tformat|<table|<row|<cell|a<rsub|11>b<rsub|11>>|<cell|\<cdots\>>|<cell|a<rsub|1n>b<rsub|1n>>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|a<rsub|m1>b<rsub|m1>>|<cell|\<cdots\>>|<cell|a<rsub|m
    n>b<rsub|m n>>>>>><matrix|<tformat|<table|<row|<cell|1>|<cell|\<cdots\>>|<cell|1>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|1>|<cell|\<cdots\>>|<cell|1>>>>>>>|<row|<cell|>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|c<rsub|1>a<rsub|11>b<rsub|11>>|<cell|\<cdots\>>|<cell|c<rsub|1>a<rsub|1n>b<rsub|1n>>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|c<rsub|m>a<rsub|m1>b<rsub|m1>>|<cell|\<cdots\>>|<cell|c<rsub|m>a<rsub|m
    n>b<rsub|m n>>>>>><matrix|<tformat|<table|<row|<cell|1>|<cell|\<cdots\>>|<cell|1>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|1>|<cell|\<cdots\>>|<cell|1>>>>>>>|<row|<cell|>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|c<rsub|1>a<rsub|11>b<rsub|11>+\<cdots\>+c<rsub|1>a<rsub|1n>b<rsub|1n>>|<cell|\<cdots\>>|<cell|c<rsub|1>a<rsub|11>b<rsub|11>+\<cdots\>+c<rsub|1>a<rsub|1n>b<rsub|1n>>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|c<rsub|m>a<rsub|m1>b<rsub|m1>+\<cdots\>+c<rsub|m>a<rsub|m
    n>b<rsub|m n>>|<cell|\<cdots\>>|<cell|c<rsub|m>a<rsub|m1>b<rsub|m1>+\<cdots\>+c<rsub|m>a<rsub|m
    n>b<rsub|m n>>>>>>>>>>
  </eqnarray*>

  RHS:

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|<matrix|<tformat|<table|<row|<cell|c<rsub|1>>|<cell|>|<cell|0>>|<row|<cell|>|<cell|\<ddots\>>|<cell|>>|<row|<cell|0>|<cell|>|<cell|c<rsub|m>>>>>><matrix|<tformat|<table|<row|<cell|a<rsub|11>>|<cell|\<cdots\>>|<cell|a<rsub|1n>>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|a<rsub|m1>>|<cell|\<cdots\>>|<cell|a<rsub|m
    n>>>>>><matrix|<tformat|<table|<row|<cell|b<rsub|11>>|<cell|\<cdots\>>|<cell|b<rsub|m1>>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|b<rsub|1n>>|<cell|\<cdots\>>|<cell|b<rsub|m
    n>>>>>>>>|<row|<cell|>|<cell|=>|<cell|<matrix|<tformat|<table|<row|<cell|c<rsub|1>>|<cell|>|<cell|0>>|<row|<cell|>|<cell|\<ddots\>>|<cell|>>|<row|<cell|0>|<cell|>|<cell|c<rsub|m>>>>>><matrix|<tformat|<table|<row|<cell|a<rsub|11>b<rsub|11>+\<cdots\>+a<rsub|1n>b<rsub|1n>>|<cell|\<cdots\>>|<cell|>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|>|<cell|\<cdots\>>|<cell|a<rsub|m1>b<rsub|m1>+\<cdots\>+a<rsub|m
    n>b<rsub|m n>>>>>>>>>>
  </eqnarray*>

  This is correct.

  When <math|diag<around*|(|c|)>=I>,\ 

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|Tr<around*|(|<with|font-series|bold|1><rsub|n\<times\>m><around*|(|A\<odot\>B|)>|)>>|<cell|=>|<cell|Tr<around*|(|B
    A<rsup|T>|)>=Tr<around*|(|A<rsup|T> B|)>>>>>>
  </equation*>

  <section|Comparing results against JAX autodiff>

  <\bibliography|bib|tm-plain|multinomial_logreg>
    <\bib-list|1>
      <bibitem*|1><label|bib-minka2000old>Thomas<nbsp>P Minka. <newblock>Old
      and new matrix algebra useful for statistics.
      <newblock><with|font-shape|italic|See www. stat. cmu.
      edu/minka/papers/matrix. html>, 4, 2000.<newblock>
    </bib-list>
  </bibliography>

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
    <associate|auto-10|<tuple|6|?>>
    <associate|auto-11|<tuple|6|?>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-3|<tuple|3|2>>
    <associate|auto-4|<tuple|3.1|2>>
    <associate|auto-5|<tuple|3.2|2>>
    <associate|auto-6|<tuple|4|2>>
    <associate|auto-7|<tuple|4.1|?>>
    <associate|auto-8|<tuple|4.2|?>>
    <associate|auto-9|<tuple|5|?>>
    <associate|bib-minka2000old|<tuple|1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      minka2000old

      minka2000old
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Introduction>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Model
      definition> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Derivative
      with respect to <with|mode|<quote|math>|vec<around*|(|W|)>>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <with|par-left|<quote|1tab>|3.1<space|2spc>Scalar Calculus
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|3.2<space|2spc>Vector Calculus
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Hessian
      with respect to <with|mode|<quote|math>|vec<around*|(|W|)>>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <with|par-left|<quote|1tab>|4.1<space|2spc>Scalar calculus
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <with|par-left|<quote|1tab>|4.2<space|2spc>Vector calculus
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Derivative
      with respect to <with|mode|<quote|math>|<with|font-series|<quote|bold>|W>>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|6<space|2spc>Comparing
      results against JAX autodiff> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>