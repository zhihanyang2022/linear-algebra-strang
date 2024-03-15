<TeXmacs|2.1.1>

<style|generic>

<\body>
  Functions (assuming nice properties) also form a vector space. And once we
  define a dot product and a norm, we form a Banach space.\ 

  <section|Dot product and norm for functions>

  Let <math|u<around*|(|x|)>> and <math|v<around*|(|x|)>> be functions
  defined on <math|<around*|[|0,1|]>>. We define the dot product to be

  <\equation*>
    u\<cdot\>v\<triangleq\><big|int><rsub|0><rsup|1>u<around*|(|x|)>v<around*|(|x|)>
    d x.
  </equation*>

  We then define the norm to be

  <\equation*>
    <around*|\<\|\|\>|u|\<\|\|\>>=<sqrt|u\<cdot\>u>
  </equation*>

  <section|Example 1>

  Let <math|u<around*|(|x|)>> be a function defined on
  <math|<around*|[|0,1|]>>. Define <math|f> to be

  <\equation*>
    f<around*|(|u|)>=<big|int><rsub|0><rsup|1>sin<around*|(|u<around*|(|x|)>|)>
    d x.
  </equation*>

  Linearize <math|f>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|d f>|<cell|=>|<cell|f<around*|(|u+d
    u|)>-f<around*|(|u|)>>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|0><rsup|1><around*|<left|[|3>|sin<around*|(|u<around*|(|x|)>+d
    u<around*|(|x|)>|)>-sin<around*|(|u<around*|(|x|)>|)>|<right|]|3>> d
    x>>>>
  </eqnarray*>

  Treating <math|u<around*|(|x|)>> as a variable, we know that

  <\eqnarray*>
    <tformat|<table|<row|<cell|sin<around*|(|u<around*|(|x|)>+d
    u<around*|(|x|)>|)>-sin<around*|(|u<around*|(|x|)>|)>>|<cell|=>|<cell|cos<around*|(|u<around*|(|x|)>|)>
    d u<around*|(|x|)>>>|<row|<cell|>|<cell|\<Downarrow\>>|<cell|>>|<row|<cell|sin<around*|(|u<around*|(|x|)>+d
    u<around*|(|x|)>|)>>|<cell|=>|<cell|sin<around*|(|u<around*|(|x|)>|)>+cos<around*|(|u<around*|(|x|)>|)>
    d u<around*|(|x|)>.>>>>
  </eqnarray*>

  Using the result, we obtain

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|0><rsup|1><around*|<left|[|3>|cos<around*|(|u<around*|(|x|)>|)>
    d u<around*|(|x|)>|<right|]|3>> d x>>|<row|<cell|>|<cell|=>|<cell|cos<around*|(|u|)>\<cdot\>d
    u>>|<row|<cell|>|<cell|\<Downarrow\>>|<cell|>>|<row|<cell|\<nabla\>f>|<cell|=>|<cell|cos<around*|(|u|)>>>>>
  </eqnarray*>

  We see that in order for <math|\<nabla\>f> to be zero, <math|u> needs to be
  a straight line <math|y=90\<pm\>360n> for <math|n\<in\>\<bbb-Z\>> (maxima)
  or <math|y=270\<pm\>360n> for <math|n\<in\>\<bbb-Z\>> (minima).\ 

  <section|Example 2: arc length formula>

  <\equation*>
    f<around*|(|u|)>=<big|int><rsub|a><rsup|b><sqrt|1+u<rprime|'><around*|(|x|)><rsup|2>>
    d x
  </equation*>

  Linearize <math|f>:

  <\eqnarray*>
    <tformat|<table|<row|<cell|d f>|<cell|=>|<cell|f<around*|(|u+d
    u|)>-f<around*|(|u|)>>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|a><rsup|b><sqrt|1+<around*|(|u<around*|(|x|)>+d
    u<around*|(|x|)>|)><rprime|'><rsup|2>>-<sqrt|1+u<rprime|'><around*|(|x|)><rsup|2>>d
    x>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|a><rsup|b><sqrt|1+<around*|(|u<rprime|'><around*|(|x|)>+d
    u<rprime|'><around*|(|x|)>|)><rsup|2>>-<sqrt|1+u<rprime|'><around*|(|x|)><rsup|2>>d
    x<space|1em><around*|(|<text|linearity of derivative operator>|)>>>>>
  </eqnarray*>

  Treating <math|u<rprime|'><around*|(|x|)>> as a variable, we know that

  <\eqnarray*>
    <tformat|<table|<row|<cell|<sqrt|1+<around*|(|u<rprime|'><around*|(|x|)>+d
    u<rprime|'><around*|(|x|)>|)><rprime|'><rsup|2>>-<sqrt|1+u<rprime|'><around*|(|x|)><rsup|2>>>|<cell|=>|<cell|<frac|1|2><around*|(|1+<around*|(|u<around*|(|x|)>|)><rprime|'><rsup|2>|)><rsup|-1/2><around*|(|2<around*|(|u<around*|(|x|)>|)><rprime|'>|)>
    d u<rprime|'><around*|(|x|)>>>|<row|<cell|>|<cell|=>|<cell|<frac|u<rprime|'><around*|(|x|)><rsup|2>|<sqrt|1+u<rprime|'><around*|(|x|)><rsup|2>>>
    d u<rprime|'><around*|(|x|)>>>>>
  </eqnarray*>

  Using this result, we obtain

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|a><rsup|b><around*|[|<frac|u<rprime|'><around*|(|x|)>|<sqrt|1+u<rprime|'><around*|(|x|)><rsup|2>>>
    d u<rprime|'><around*|(|x|)>|]> d x>>>>
  </eqnarray*>

  But this is not in terms of <math|d u>. Reviewing intergration by parts

  \;

  Integration by parts

  <\equation*>
    <big|int><rsub|a><rsup|b>f<around*|(|x|)>g<rprime|'><around*|(|x|)>=f<around*|(|x|)>g<around*|(|x|)><around*|<left|\||3>|<rsub|a><rsup|b>|<right|.>>-<big|int><rsub|a><rsup|b>f<rprime|'><around*|(|x|)>g<around*|(|x|)>
    d x
  </equation*>

  Apply integration by parts, we get (\<ast\>)

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|=>|<cell|<frac|u<rprime|'><around*|(|x|)>|<sqrt|1+u<rprime|'><around*|(|x|)><rsup|2>>>
    d u<around*|(|x|)><around*|<left|\||3>|<rsub|a><rsup|b>|<right|.>>-<big|int><around*|[|<frac|u<rprime|''><around*|(|x|)>|1+u<rprime|'><around*|(|x|)><rsup|3/2>>
    d u<around*|(|x|)>|]>d x>>>>
  </eqnarray*>

  Suppose we fix <math|u<around*|(|a|)>=A,u<around*|(|b|)>=B> (also want
  <math|d u<around*|(|a|)>=d u<around*|(|b|)>=0>; perturbations with end
  points fixed), first term becomes zero.

  Obviously, the gradient would be

  <\equation*>
    \<nabla\>f<around*|(|x|)>=<frac|u<rprime|''><around*|(|x|)>|1+u<rprime|'><around*|(|x|)><rsup|3/2>>
  </equation*>

  (\<ast\>) Differentiate the first part with respect to <math|x>:

  <\equation*>
    <frac|u<rprime|''><around*|(|x|)><sqrt|1+u<rprime|'><around*|(|x|)><rsup|2>>-u<rprime|'><around*|(|x|)><frac|1|2><around*|(|1+u<rprime|'><around*|(|x|)><rsup|2>|)><rsup|-1/2>2u<rprime|'><around*|(|x|)>u<rprime|''><around*|(|x|)>|1+u<rprime|'><around*|(|x|)><rsup|2>>
  </equation*>

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|=>|<cell|<frac|u<rprime|''><around*|(|x|)><around*|(|1+u<rprime|'><around*|(|x|)><rsup|2>|)><rsup|1/2>-u<rprime|'><around*|(|x|)><rsup|2>u<rprime|''><around*|(|x|)><around*|(|1+u<rprime|'><around*|(|x|)><rsup|2>|)><rsup|-1/2>|1+u<rprime|'><around*|(|x|)><rsup|2>>>>|<row|<cell|>|<cell|=>|<cell|<frac|u<rprime|''><around*|(|x|)><around*|(|1+u<rprime|'><around*|(|x|)><rsup|2>|)>-u<rprime|'><around*|(|x|)><rsup|2>u<rprime|''><around*|(|x|)>|1+u<rprime|'><around*|(|x|)><rsup|3/2>>>>|<row|<cell|>|<cell|=>|<cell|<frac|u<rprime|''><around*|(|x|)>|1+u<rprime|'><around*|(|x|)><rsup|3/2>>>>>>
  </eqnarray*>

  \;

  <section|Generalization to Euler's formula>

  <\equation*>
    f<around*|(|u|)>=<big|int><rsub|a><rsup|b>F<around*|(|u<around*|(|x|)>,u<rprime|'><around*|(|x|)>,x|)>
    d x
  </equation*>

  Linearize

  <\eqnarray*>
    <tformat|<table|<row|<cell|d f>|<cell|=>|<cell|<big|int><rsub|a><rsup|b><around*|<left|[|3>|F<around*|(|u<around*|(|x|)>+d
    u<around*|(|x|)>,u<rprime|'><around*|(|x|)>+d
    u<rprime|'><around*|(|x|)>,x|)>-F<around*|(|u<around*|(|x|)>,u<rprime|'><around*|(|x|)>,x|)>|<right|]|3>>
    d x>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|a><rsup|b><around*|<left|[|1>|<frac|\<partial\>F|\<partial\>u>d
    u<around*|(|x|)>+<frac|\<partial\>F|\<partial\>u<rprime|'>>d
    u<rprime|'><around*|(|x|)>|<right|]|1>>d
    x>>|<row|<cell|>|<cell|>|<cell|<around*|(|<text|apply integration by
    parts to the second term only>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|a><rsup|b><around*|<left|[|1>|<frac|\<partial\>F|\<partial\>u>u<around*|(|x|)>-<around*|(|<frac|\<partial\>F|\<partial\>u<rprime|'>>|)><rprime|'>d
    u<around*|(|x|)>|<right|]|1>>d x+<frac|\<partial\>F|\<partial\>u<rprime|'>>d
    u<around*|(|x|)><around*|<left|\||3>|<rsub|a><rsup|b>|<right|.>>>>|<row|<cell|>|<cell|=>|<cell|>>>>
  </eqnarray*>

  Example here lol

  \;

  <\equation*>
    F<around*|(|u<around*|(|x|)>+d u<around*|(|x|)>,u<rprime|'><around*|(|x|)>+d
    u<rprime|'><around*|(|x|)>,x|)>-F<around*|(|u<around*|(|x|)>,u<rprime|'><around*|(|x|)>,x|)>=<frac|\<partial\>F|\<partial\>u>d
    u<around*|(|x|)>+<frac|\<partial\>F|\<partial\>u<rprime|'>>d
    u<rprime|'><around*|(|x|)>
  </equation*>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1|../../../../.TeXmacs/texts/scratch/no_name_29.tm>>
    <associate|auto-2|<tuple|2|1|../../../../.TeXmacs/texts/scratch/no_name_29.tm>>
    <associate|auto-3|<tuple|3|1|../../../../.TeXmacs/texts/scratch/no_name_29.tm>>
    <associate|auto-4|<tuple|4|?|../../../../.TeXmacs/texts/scratch/no_name_29.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Dot
      product and norm for functions> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Example
      1> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Example
      2: arc length formula> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>