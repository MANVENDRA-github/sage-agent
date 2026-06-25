<!--
Blog draft for sage-agent.

Title options (pick one):
  - I didn't want to build a memory agent. I wanted to know if it worked.
  - The agent was the easy part. Trusting it wasn't.
  - How I learned to stop trusting AI demos and grade my own

Meta description (Dev.to / Hashnode): I built a memory agent on a free LLM,
then built a harness to grade it on 84 cases. The grading taught me more than
the agent did.

Tags: ai, llm, langgraph, python, machinelearning

Status: draft. Personalize the opening itch and the takeaways with a real
moment from the build before publishing.
-->

# I didn't want to build a memory agent. I wanted to know if it worked.

I didn't set out to build a memory agent. I set out to answer a question that bugged me every time I saw one of these demos: how does anyone actually know it works? A name echoed back next session proves nothing. It worked once, on the one input the person filming the demo chose to type.

The cases I actually cared about were the awkward ones. What happens when someone tells you they drive a Camry, and three weeks later says they switched to a Tesla? What happens when they ask about something they never told you? What happens when they answer their own question halfway through asking it? You don't see any of that in a screenshot, and that's exactly where a memory agent either earns its keep or quietly falls apart.

So I built one myself, and then I built the thing I actually wanted, which was a way to grade it honestly. The project is called **sage-agent**. The agent is the part people will click on. The grader is the part I'm proud of.

> Live demo: [sage-agent.streamlit.app](https://sage-agent.streamlit.app/) · Code: [github.com/MANVENDRA-github/sage-agent](https://github.com/MANVENDRA-github/sage-agent)

## What I built, before I get to the interesting part

I started from LangChain's [`memory-agent`](https://github.com/langchain-ai/memory-agent) template and rebuilt the pieces that mattered to me. What I ended up with is a LangGraph loop with four tools the model can reach for on its own: one to recall what you told it before, one to save something new, one to search the web for things I never told it, and one to track goals separately from ordinary facts.

I gave myself one rule for the whole project: it had to run for free. A free model through OpenRouter, embeddings computed locally on my machine, no paid keys anywhere. I didn't want the first sentence of my README to be "first, add a credit card." Anyone can clone it and try it.

That's the agent. Honestly, wiring four tools into a loop is not the hard part, and it's not what I want to talk about. I want to talk about the moment I stopped trusting it and started measuring it.

## The first thing measuring taught me: let it choose

My early version searched memory before every single reply. Every turn, whether it made sense or not. It felt safe. It was also a little stupid, because half the time the question is "what's 2 plus 2" and there is nothing in your past worth looking up.

So I took that forced step out and turned recall into a tool the model decides to use. Now it searches your memory when the question is about you, searches the web when it's about the world, and just answers when it already knows.

The thing I was scared of was obvious: what if it just stops looking things up and starts making stuff up? That fear is the whole reason the grader exists. On the cases where the answer was sitting right there in memory, the model chose to go look 100% of the time. I didn't have to take that on faith. I had a number.

## The decision the grader forced on me

This is my favourite story from the whole build, because the test changed the code rather than the other way around.

Go back to the Camry and the Tesla. When you tell the agent you switched cars, you should end up with one memory, not two. Inside, that's a replace. And the easy, obvious way to do a replace is to find the old record and overwrite it where it sits.

Here's the catch I didn't see coming. My grader decides whether the agent saved anything by looking for a new memory that wasn't there a moment ago. An overwrite-in-place doesn't create anything new. So the grader would watch the agent do exactly the right thing, resolve the contradiction perfectly, and still mark it down as "decided not to save." My score would punish correct behaviour.

So I changed the implementation. A replace now deletes the old memory and writes a fresh one. The update finally counts as the save it really is, and the number tells the truth. I would have shipped the wrong version and never known, if I hadn't been scoring it.

The reward for all this fussing is one number I trust: the agent's save decisions land at an F1 of 0.983, and that number didn't move even after I tore out the control flow and rebuilt it as the tool-calling loop. There's exactly one case it still gets wrong, and I can tell you its ID. That's the whole reason I did this. I wanted to be able to point at the one thing that's broken instead of waving at a demo and saying "feels good."

## The number I'm proudest of isn't 100%

The harder test is the second one. With four tools sitting in front of it, does the model reach for the right one, and does it know when the right move is to do nothing at all?

My first version of that test scored a clean 100%, so I deleted it. A test everything passes isn't testing anything. I rewrote it to be mean on purpose, including cases where the correct answer is to keep your hands off the keyboard.

Now it scores somewhere between 91% and 94% depending on the run, and the failures are the most useful thing in the whole repo:

- Someone says *"I finally did it, mark my goal as done!"* but they've got two active goals saved. The right move is to ask which one. Instead the model just closes one and hopes. It gets this wrong every single time.
- Someone says *"remind me what my name is. Oh wait, never mind, it's Alex."* They answered themselves. There is nothing to do. The model saves it anyway.

Both of those are the same flaw wearing two outfits: this model would rather act than ask, and would rather act than wait. I couldn't have fixed an instinct I couldn't see. Now it's two rows in a table with names on them.

One more honest note, because it matters. This free model isn't deterministic even with the temperature pinned at zero. Run the suite three times, get three slightly different scores. So I don't report one number. I report the range across three runs. A single clean figure on a model that wobbles is a marketing slide, not a measurement.

## What I actually took away from this

If you're building one of these, here's what I'd save you:

Write the grader early, not at the end. The agent is the weekend. Knowing whether it works is the month, and if you leave it for last you've already trusted a hundred things you can't see.

Let the score argue with your design. The delete-and-rewrite thing wasn't me being clever. The grader told me my code was wrong and I listened.

And be suspicious of a 100%. Nearly every time, it means your test is too soft, not that your code is perfect.

The agent is maybe 40% of this repo. The grader is the rest, and it's the only part that taught me anything. What's next is letting old memories fade, having the agent summarise what it knows about you, and running the same tests against a stronger model to find out whether these numbers are the agent being good or just the free one getting lucky.

If you want to prod at it, the [demo](https://sage-agent.streamlit.app/) and the [code](https://github.com/MANVENDRA-github/sage-agent) are both live, and every number in this post comes from a results file committed in the repo. No screenshots, no trust required.
